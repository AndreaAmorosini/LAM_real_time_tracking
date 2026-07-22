declare module "gaussian-splat-renderer-for-lam" {
  export class GaussianSplatRenderer {
    static getInstance(
      container: HTMLElement,
      assetPath: string,
      options?: {
        getExpressionData?: () => Record<string, number>;
        getChatState?: () => unknown;
        downloadProgress?: (value: number) => void;
        loadProgress?: (value: number) => void;
        backgroundColor?: string;
        alpha?: number;
      }
    ): Promise<any>;
  }
}
