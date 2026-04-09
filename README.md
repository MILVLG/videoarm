<div align="center">
<h1>VideoARM: Agentic Reasoning over Hierarchical Memory for Long-Form Video Understanding</h1>

  **[School of Computer Science, Hangzhou Dianzi University, China](https://www.hdu.edu.cn)**

  [Yufei Yin](https://yinyf0804.github.io), [Qianke Meng](https://qiankemeng.github.io), [Minghao Chen](https://faculty.hdu.edu.cn/jsjxy/cmh2/main.htm), [Jiajun Ding](https://mil.hdu.edu.cn/people/jiajun_ding/index.html), [Zhenwei Shao](https://scholar.google.com/citations?user=j87m-woAAAAJ&hl=en), [Zhou Yu](https://faculty.hdu.edu.cn/jsjxy/yz/main.htm)<sup>*</sup>
</div>

[![arXiv](https://img.shields.io/badge/arXiv-2512.12360-b31b1b.svg)](https://arxiv.org/abs/2512.12360)
[![GitHub](https://img.shields.io/github/stars/MILVLG/videoarm?style=social)](https://github.com/MILVLG/videoarm)


```bibtex
@inproceedings{yin2026videoarm,
  title={VideoARM: Agentic Reasoning over Hierarchical Memory for Long-Form Video Understanding},
  author={Yin, Yufei and Meng, Qianke and Chen, Minghao and Ding, Jiajun and Shao, Zhenwei and Yu, Zhou},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2026}
}
```
## Installation

```bash
pip install -e .
```

Or install dependencies directly:

```bash
pip install opencv-python openai requests python-dotenv numpy
```

### API keys

Copy `.env.example` to `.env` and fill in your OpenAI API key:

```
OPENAI_API_KEY=sk-...
OPENAI_BASE_URL=https://api.openai.com/v1
```

---

## Usage

### Command line

```bash
# Open-ended question
python main.py --video path/to/video.mp4 --question "What happens in this video?"

# Multiple-choice (letter A-D, default)
python main.py --video video.mp4 --question "A. ... B. ... C. ... D. ..." --multiple-choice

# Multiple-choice (number 0-4)
python main.py --video video.mp4 --question "..." --multiple-choice --choice-format number

# Override the controller model
python main.py --video video.mp4 --question "..." --model gpt-4o

# Run without saving the result trace
python main.py --video video.mp4 --question "..." --no-save
```

## Configuration

### Model selection

| Environment variable | Default | Description |
|---|---|---|
| `VIDEOARM_MODEL_CONTROLLER` | `o3` | Reasoning controller |
| `VIDEOARM_MODEL_CLIP_ANALYZER` | `gpt-4.1` | Clip Analyzer + Scene Snapper |
| `VIDEOARM_MODEL_AUDIO_TRANSCRIBER` | `whisper-1` | Audio Transcriber |

### Pipeline parameters

| Environment variable | Default | Description |
|---|---|---|
| `VIDEOARM_MAX_ITERATIONS` | `10` | Step budget N |
| `VIDEOARM_MAX_FRAMES_PER_TOOL` | `150` | Max frames passed per tool call |
| `VIDEOARM_FRAME_ANALYSIS_MAX_FRAMES` | `50` | Frames sampled by Clip Analyzer |
| `VIDEOARM_AUDIO_MAX_FRAMES` | `15000` | Max frames for audio extraction |

### Per-component API overrides

To route different tools to different API endpoints:

```
VIDEOARM_API_KEY_CONTROLLER=sk-...
VIDEOARM_BASE_URL_CONTROLLER=https://...

VIDEOARM_API_KEY_CLIP_ANALYZER=sk-...
VIDEOARM_BASE_URL_CLIP_ANALYZER=https://...
```

See `.env.example` for the full list of options.



