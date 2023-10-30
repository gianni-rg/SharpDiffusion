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

// Ported/based on Hugging Face Diffusers library
// (https://github.com/huggingface/diffusers)
// Copyright (C) 2022-2023 The HuggingFace Inc. team.
// Licensed under the Apache License, Version 2.0.

namespace SharpDiffusion;

/// <summary>
/// Output class for Stable Diffusion pipelines
/// </summary>
public class StableDiffusionPipelineOutput : BaseOutput
{
    /// <summary>
    /// List of denoised images of length 'batchSize'. Images array present the denoised images of the diffusion pipeline.
    /// </summary>
    public List<Image<Rgba32>> Images { get; private set; }

    /// <summary>
    /// List of flags denoting whether the corresponding generated image likely represents "Not Safe For Work" (NSFW) content, or <c>null</c> if safety checking could not be performed.
    /// </summary>
    public List<bool> NSFWContentDetected { get; private set; }

    public StableDiffusionPipelineOutput(List<Image<Rgba32>> images, List<bool> nsfwContentDetected)
    {
        Images = images;
        NSFWContentDetected = nsfwContentDetected;
    }
}
