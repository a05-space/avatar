import defaultBsData from "../asset/expression.json"
import * as GaussianSplats3D from "gaussian-splat-renderer-for-lam"

export class GaussianAvatar {
  private _avatarDivEle: HTMLDivElement;
  private _assetsPath = "";
  public curState = "Idle";
  private _renderer: any;
  private _expressionData: any = defaultBsData;
  private _talking = false;

  constructor(container: HTMLDivElement, assetsPath: string) {
    this._avatarDivEle = container;
    this._assetsPath = assetsPath;
    this._init();
  }

  private _init() {
    if (!this._avatarDivEle || !this._assetsPath) {
      throw new Error("Lack of necessary initialization parameters");
    }
  }

  public start() {
    this.render();
  }

  public setExpressionData(data: any) {
    if (!data || !data["frames"] || !data["names"]) {
      return;
    }
    this._expressionData = data;
    this.startTime = performance.now() / 1000;
  }

  public setTalking(talking: boolean) {
    this._talking = talking;
    this.curState = talking ? "Responding" : "Idle";
  }

  public async render() {
    this._renderer = await GaussianSplats3D.GaussianSplatRenderer.getInstance(
      this._avatarDivEle,
      this._assetsPath,
      {
        getChatState: this.getChatState.bind(this),
        getExpressionData: this.getArkitFaceFrame.bind(this),
        backgroundColor: "0x000000",
        alpha: 0.1
      },
    );
    this.startTime = performance.now() / 1000;
  }

  startTime = 0;

  public getChatState() {
    return this.curState;
  }

  public getArkitFaceFrame() {
    const result: any = {};

    if (!this._talking) {
      this._expressionData["names"].forEach((name: string) => {
        result[name] = 0;
      });
      return result;
    }

    const length = this._expressionData["frames"].length;
    const frameInfoInternal = 1.0 / 30.0;
    const currentTime = performance.now() / 1000;
    const calcDelta = (currentTime - this.startTime) % (length * frameInfoInternal);
    const frameIndex = Math.floor(calcDelta / frameInfoInternal);

    this._expressionData["names"].forEach((name: string, index: number) => {
      result[name] = this._expressionData["frames"][frameIndex]["weights"][index];
    });
    return result;
  }
}
