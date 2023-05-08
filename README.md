# SharpDiffusion

This is an *experimental* .NET implementation of the [diffusers](https://github.com/huggingface/diffusers) Python library from Huggingface.

- [SharpDiffusion](#sharpdiffusion)
  - [Introduction](#introduction)
  - [Getting Started](#getting-started)
    - [Project Organization](#project-organization)
    - [Setup a local copy](#setup-a-local-copy)
    - [Minimal usage sample](#minimal-usage-sample)
  - [Contribution](#contribution)
  - [License](#license)

## Introduction

**It is an on-going work in progress, built in my spare time for fun & learning.**

Currently it's only a *very basic, bare-bone, partial porting of the original library*.  
Many of the features are not there yet (i.e. safety check model is not supported), and it only works using the original *Stable Diffusion v1.x ONNX models*.

> You can find an example application using this library in this project: [Generative AI .NET Playground](https://github.com/gianni-rg/gen-ai-net-playground)

You have to get the models from Hugging Face:

- [Stable Diffusion Models v1.4](https://huggingface.co/CompVis/stable-diffusion-v1-4)
- [Stable Diffusion Models v1.5](https://huggingface.co/runwayml/stable-diffusion-v1-5)

Once you have selected a model version repository, click `Files and Versions`, then select the `ONNX branch`. Clone the repository (you need Git LFS to get the pre-trained model weights). Once cloned, set the proper path to load the models from, using `OnnxStableDiffusionPipeline.FromPretrained`. The folders that will be used are: `unet`, `vae_decoder`, `text_encoder`.

For the tokenizer model ([CLIP Tokenizer](https://huggingface.co/docs/transformers/model_doc/clip)), currently the library leverages the implementation provided by Microsoft in the [ONNX Runtime Extensions](https://github.com/microsoft/onnxruntime-extensions). You can get the pre-trained ONNX model from the [ONNX Runtime Extensions .NET Samples](https://github.com/microsoft/onnxruntime-extensions/tree/main/tutorials/demo4dotnet/ClipTokenizerTest) folder: copy it in the `tokenizer` folder, along with the other models, and rename it as `model.onnx`.

---

## Getting Started

### Project Organization

    ├── LICENSE
    ├── README.md                      <- The top-level README for developers using this project
    ├── docs                           <- Project documentation
    ├── src                            <- Source code
    │   ├── SharpDiffusion             <- Core library
    |
    └── ...                            <- other files

### Setup a local copy

Clone the repository and build. You should be able to generate the library and use it in your own projects.

### Minimal usage sample

```csharp
var options = new Dictionary<string, string> {
    { "device_id", "0"},
    { "gpu_mem_limit",  "15000000000" }, // 15GB
    { "arena_extend_strategy", "kSameAsRequested" },
};

var modelId = "PATH-TO-ONNX-MODELS-FOLDER";
var provider = "CUDAExecutionProvider"; // "CPUExecutionProvider";

var sdPipeline = OnnxStableDiffusionPipeline.FromPretrained(modelId, provider: provider, sessionOptions: options);

var sdConfig = new StableDiffusionConfig
{
    NumInferenceSteps = 20,
    GuidanceScale = 7.5,
    NumImagesPerPrompt = 1,
};

var prompts = new List<string> {
    "PROMPT",
};

var negativePrompts = new List<string> {
    string.Empty,
};

sdPipeline.Run(prompts, negativePrompts, sdConfig);
```

---

## Contribution

The project is constantly evolving and contributions are warmly welcomed.

I'm more than happy to receive any kind of contribution to this experimental project: from helpful feedbacks to bug reports, documentation, usage examples, feature requests, or directly code contribution for bug fixes and new and/or improved features.

Feel free to file issues and pull requests on the repository and I'll address them as much as I can, *with a best effort approach during my spare time*.

> Development is mainly done on Windows, so other platforms are not directly developed, tested or supported.  
> An help is kindly appreciated in make the application work on other platforms as well.

## License

You may find specific license information for third party software in the [third-party-programs.txt](./third-party-programs.txt) file.  
Where not otherwise specified, everything is licensed under the [APACHE 2.0 License](./LICENSE).

Copyright (C) 2022-2023 Gianni Rosa Gallina.
