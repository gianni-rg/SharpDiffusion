// Copyright (C) 2023 Gianni Rosa Gallina. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License").
// You may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0.
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

namespace SharpDiffusion;

public class StableDiffusionConfig
{
    // Default settings
    public int NumInferenceSteps = 15;
    public double GuidanceScale = 7.5;
    public int Height = 512;
    public int Width = 512;

    //List<string>? negativePrompt = null,
    //float eta = 0.0f,
    //object? generator = null, // TODO: np.random.RandomState, to be ported from Python
    //Array? latents = null,
    //string outputType = "pil", // TODO: to be ported from Python
    //bool returnDict = true,
    //Action<int, int, Array>? callback = null,

    public string OrtExtensionsPath = "ortextensions.dll";
    public string TokenizerOnnxPath = "cliptokenizer.onnx";
    public string TextEncoderOnnxPath = "";
    public string UnetOnnxPath = "";
    public string VaeDecoderOnnxPath = "";
    public string SafetyModelPath = "";

    public int? Seed = null;
    public int Channels = 4;
    public int NumImagesPerPrompt = 1;

    public int CallbackSteps = 1;
    public int TokenizerModelMaxLength = 77;
    public int ModelEmbeddingSize = 768;
}
