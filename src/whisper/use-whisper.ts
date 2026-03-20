import { Whisper } from "@client-ai/whisper/whisper";
import OpenAI from "openai";
import { useRef } from "react";
const modelPath = process.env.WHISPER_MODEL_PATH;

export const useWhisper = () => {
  const whisper = useRef<Whisper | null>(null);

  const localRun = async (float32Array: Float32Array) => {
    if (!whisper.current) {
      whisper.current = new Whisper(modelPath!);
    }
    await whisper.current.ready();
    const result = await whisper.current.run(float32Array);
    return result.str.data[0] as string;
  };

  const remoteRun = async (blob: Blob) => {
    function getBaseUrl() {
      return typeof window !== "undefined" ? window.location.origin : "";
    }
    const baseURL = `${getBaseUrl()}/api`;

    const openai = new OpenAI({
      baseURL,
      apiKey: "",
      dangerouslyAllowBrowser: true,
    });
    const data = await openai.audio.transcriptions.create({
      file: new File([blob], "audio.wav"),
      model: "whisper-1",
    });
    return data.text;
  };
  const run = async <T extends boolean>(
    useLocal: T,
    input: T extends true ? Float32Array : Blob,
  ) => {
    if (useLocal) {
      if (!(input instanceof Float32Array)) {
        throw new Error("Local mode requires Float32Array input");
      }
      return localRun(input);
    } else {
      if (!(input instanceof Blob)) {
        throw new Error("Remote mode requires Blob input");
      }
      return remoteRun(input);
    }
  };

  return { localRun, remoteRun, run };
};
