import { GaussianAvatar } from './gaussianAvatar';

const container = document.getElementById('WebRender') as HTMLDivElement | null;
const assetPath = './asset/arkit/default.zip';

if (container) {
  const gaussianAvatar = new GaussianAvatar(container, assetPath);
  gaussianAvatar.start();

  const renderInfo = document.getElementById('render-info') as HTMLSpanElement | null;

  const audioInput = document.getElementById('audio-input') as HTMLInputElement | null;
  const uploadBtn = document.getElementById('upload-btn') as HTMLButtonElement | null;
  const audioPlayer = document.getElementById('audio-player') as HTMLAudioElement | null;
  const textInput = document.getElementById('text-input') as HTMLInputElement | null;
  const ttsBtn = document.getElementById('text2audio-btn') as HTMLButtonElement | null;
  const t2aBtn = document.getElementById('text2avatar-btn') as HTMLButtonElement | null;

  const apiBase = 'http://localhost:8109';

  if (audioInput && uploadBtn) {
    uploadBtn.onclick = async () => {
      if (!audioInput.files || audioInput.files.length === 0) {
        return;
      }

      const file = audioInput.files[0];
      const formData = new FormData();
      formData.append('file', file);

      try {
        const resp = await fetch(`${apiBase}/api/audio2avatar`, {
          method: 'POST',
          body: formData,
        });

        if (!resp.ok) {
          console.error('Audio upload failed', await resp.text());
          return;
        }

        const respData: any = await resp.json();
        const expressionData = respData && respData.expression ? respData.expression : respData;
        const audioUrl = typeof respData?.audioUrl === 'string' ? respData.audioUrl : null;
        const audioBase64 = typeof respData?.audioBase64 === 'string' ? respData.audioBase64 : null;

        if (audioPlayer && (audioUrl || audioBase64)) {
          if (!audioPlayer.paused) {
            audioPlayer.pause();
          }
          audioPlayer.currentTime = 0;

          if (audioUrl) {
            audioPlayer.src = audioUrl;
          } else if (audioBase64) {
            audioPlayer.src = audioBase64;
          }

          audioPlayer.onplay = () => {
            gaussianAvatar.setExpressionData(expressionData);
            gaussianAvatar.setTalking(true);
          };

          audioPlayer.onended = () => {
            gaussianAvatar.setTalking(false);
          };

          audioPlayer.play().catch((err) => {
            console.error('Audio play failed', err);
            gaussianAvatar.setExpressionData(expressionData);
            gaussianAvatar.setTalking(true);
          });
        } else {
          gaussianAvatar.setExpressionData(expressionData);
          gaussianAvatar.setTalking(true);
        }
        if (renderInfo && expressionData) {
          const frameCount = Array.isArray(expressionData?.frames) ? expressionData.frames.length : 0;
          const fps = typeof expressionData?.metadata?.fps === 'number' ? expressionData.metadata.fps : 30;
          const time = new Date().toLocaleTimeString();
          renderInfo.textContent = `HTTP 推理完成：fps=${fps}, 帧数=${frameCount}（${time}）`;
        }
      } catch (err) {
        console.error('Failed to send audio', err);
      }
    };
  }

  if (textInput && ttsBtn) {
    ttsBtn.onclick = async () => {
      const text = textInput.value.trim();
      if (!text) return;
      try {
        const resp = await fetch(`${apiBase}/api/text2audio`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ text }),
        });
        if (!resp.ok) {
          console.error('TTS failed', await resp.text());
          return;
        }
        const data: any = await resp.json();
        const segs = Array.isArray(data?.sentences) ? data.sentences.length : 0;
        if (renderInfo) {
          renderInfo.textContent = `TTS 完成：段落=${segs}，uuid=${data?.uuid || ''}`;
        }
      } catch (e) {
        console.error('TTS error', e);
      }
    };
  }

  if (textInput && t2aBtn) {
    t2aBtn.onclick = async () => {
      const text = textInput.value.trim();
      if (!text) return;
      try {
        const resp = await fetch(`${apiBase}/api/text2avatar`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ text }),
        });
        if (!resp.ok) {
          console.error('Text2Avatar failed', await resp.text());
          return;
        }
        const respData: any = await resp.json();
        const expressionData = respData && respData.expression ? respData.expression : respData;
        const audioBase64 = typeof respData?.audioBase64 === 'string' ? respData.audioBase64 : null;

        if (audioPlayer && audioBase64) {
          if (!audioPlayer.paused) {
            audioPlayer.pause();
          }
          audioPlayer.currentTime = 0;
          audioPlayer.src = audioBase64;
          audioPlayer.onplay = () => {
            gaussianAvatar.setExpressionData(expressionData);
            gaussianAvatar.setTalking(true);
          };
          audioPlayer.onended = () => {
            gaussianAvatar.setTalking(false);
          };
          audioPlayer.play().catch((err) => {
            console.error('Audio play failed', err);
            gaussianAvatar.setExpressionData(expressionData);
            gaussianAvatar.setTalking(true);
          });
        } else {
          gaussianAvatar.setExpressionData(expressionData);
          gaussianAvatar.setTalking(true);
        }
        if (renderInfo && expressionData) {
          const frameCount = Array.isArray(expressionData?.frames) ? expressionData.frames.length : 0;
          const fps = typeof expressionData?.metadata?.fps === 'number' ? expressionData.metadata.fps : 30;
          const time = new Date().toLocaleTimeString();
          renderInfo.textContent = `Text2Avatar 完成：fps=${fps}, 帧数=${frameCount}（${time}）`;
        }
      } catch (e) {
        console.error('Text2Avatar error', e);
      }
    };
  }
}
