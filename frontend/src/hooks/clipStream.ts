/**
 * Plays a Higgs fragmented-MP4 stream as the bytes arrive.
 * The first frames show before the clip has finished generating.
 */

const MIMES = [
  'video/mp4; codecs="avc1.42E01E,mp4a.40.2"',
  'video/mp4; codecs="avc1.4D401E,mp4a.40.2"',
  'video/mp4; codecs="avc1.640028,mp4a.40.2"',
];

type SeatStream = {
  seq: number;
  url: string;
  media: MediaSource;
  source: SourceBuffer | null;
  queue: Uint8Array[];
  parts: Uint8Array[];
  failed: boolean;
  shown: boolean;
  complete: boolean;
};

export function decodeB64(b64: string): Uint8Array {
  const binary = atob(b64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
  return bytes;
}

export function createClipHub(onUrl: (seat: number, url: string, seq: number) => void) {
  const seats = new Map<number, SeatStream>();

  const pump = (stream: SeatStream) => {
    const source = stream.source;
    if (!source || source.updating || stream.queue.length === 0) return;
    const chunk = stream.queue.shift();
    if (!chunk) return;
    try {
      source.appendBuffer(chunk);
    } catch {
      stream.failed = true;
      stream.queue = [];
    }
  };

  const finish = (stream: SeatStream) => {
    if (stream.media.readyState !== "open") return;
    if (stream.source?.updating || stream.queue.length > 0) return;
    try {
      stream.media.endOfStream();
    } catch {
      /* already ended */
    }
  };

  const showBlob = (seat: number, stream: SeatStream) => {
    if (!stream.parts.length) return;
    const blob = new Blob(stream.parts, { type: "video/mp4" });
    const url = URL.createObjectURL(blob);
    URL.revokeObjectURL(stream.url);
    stream.url = url;
    onUrl(seat, url, stream.seq);
  };

  return {
    open(seat: number, seq: number) {
      const previous = seats.get(seat);
      if (previous) {
        try {
          if (previous.media.readyState === "open") previous.media.endOfStream();
        } catch {
          /* already closed */
        }
        URL.revokeObjectURL(previous.url);
      }
      const media = new MediaSource();
      const stream: SeatStream = {
        seq,
        url: URL.createObjectURL(media),
        media,
        source: null,
        queue: [],
        parts: [],
        failed: false,
        shown: false,
        complete: false,
      };
      seats.set(seat, stream);
      media.addEventListener("sourceopen", () => {
        if (seats.get(seat) !== stream) return;
        const mime = MIMES.find((item) => MediaSource.isTypeSupported(item));
        if (!mime) {
          stream.failed = true;
          return;
        }
        try {
          const source = media.addSourceBuffer(mime);
          source.mode = "sequence";
          source.addEventListener("updateend", () => {
            if (!stream.shown) {
              stream.shown = true;
              onUrl(seat, stream.url, stream.seq);
            }
            pump(stream);
            if (stream.complete && stream.queue.length === 0) finish(stream);
          });
          source.addEventListener("error", () => {
            stream.failed = true;
          });
          stream.source = source;
          pump(stream);
        } catch {
          stream.failed = true;
        }
      });
    },

    chunk(seat: number, seq: number, bytes: Uint8Array) {
      const stream = seats.get(seat);
      if (!stream || stream.seq !== seq) return;
      stream.parts.push(bytes);
      if (stream.failed) return;
      stream.queue.push(bytes);
      pump(stream);
    },

    end(seat: number, seq: number) {
      const stream = seats.get(seat);
      if (!stream || stream.seq !== seq) return;
      stream.complete = true;
      if (!stream.parts.length) {
        URL.revokeObjectURL(stream.url);
        seats.delete(seat);
        return;
      }
      if (stream.failed || !stream.shown) showBlob(seat, stream);
      else if (!stream.source?.updating && stream.queue.length === 0) finish(stream);
    },

    drop(seat: number, seq: number) {
      const stream = seats.get(seat);
      if (!stream || stream.seq !== seq) return;
      URL.revokeObjectURL(stream.url);
      seats.delete(seat);
    },

    reset() {
      for (const stream of seats.values()) URL.revokeObjectURL(stream.url);
      seats.clear();
    },
  };
}
