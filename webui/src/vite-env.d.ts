/// <reference types="vite/client" />

interface ImportMetaEnv {
  /** Base URL of the model host, one folder per model, e.g. a Hugging Face repository at a tag (…/resolve/v0.1.0/); unset: models/ next to the page */
  readonly VITE_MODELS_URL?: string;
  /** Base URL of the VascX models (e.g. Eyened/vascx on Hugging Face at a revision); unset: models/vascx/ next to the page */
  readonly VITE_VASCX_URL?: string;
  /** Base URL of the RRWNet segmentation model; unset: models/rrwnet/ next to the page */
  readonly VITE_RRWNET_URL?: string;
  /** Base URL of the TPOT enhancement models; unset: models/tpot/ next to the page */
  readonly VITE_TPOT_URL?: string;
  /** Base URL of the language model for AI reports (e.g. onnx-community/Qwen3-1.7B-ONNX on Hugging Face at a revision); unset: models/llm/ next to the page */
  readonly VITE_LLM_URL?: string;
  /** The report service (the lab's API through a RetiVASC server), e.g. https://<tunnel host>/api/report; unset: api/report next to the page */
  readonly VITE_REPORT_URL?: string;
}
