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

using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

public static class TensorHelpers
{
    public static DenseTensor<TTensorType> CreateTensor<TTensorType>(TTensorType[] data, int[] dimensions)
    {
        return new DenseTensor<TTensorType>(data, dimensions);
    }

    public static DenseTensor<Float16> DivideTensorByFloat(Float16[] data, float value, int[] dimensions)
    {
        for (int i = 0; i < data.Length; i++)
        {
            data[i] = data[i].Div((Float16)value);
        }

        return CreateTensor(data, dimensions);
    }
    public static DenseTensor<float> DivideTensorByFloat(float[] data, float value, int[] dimensions)
    {
        for (int i = 0; i < data.Length; i++)
        {
            data[i] = data[i] / value;
        }

        return CreateTensor(data, dimensions);
    }

    public static DenseTensor<float> MultipleTensorByFloat(float[] data, float value, int[] dimensions)
    {
        for (int i = 0; i < data.Length; i++)
        {
            data[i] = data[i] * value;
        }

        return CreateTensor(data, dimensions);
    }

    public static DenseTensor<Float16> MultipleTensorByFloat(Float16[] data, Float16 value, int[] dimensions)
    {
        for (int i = 0; i < data.Length; i++)
        {
            data[i] = data[i].Mul(value);
        }

        return CreateTensor(data, dimensions);
    }

    public static DenseTensor<float> MultipleTensorByFloat(Tensor<float> data, float value)
    {
        return MultipleTensorByFloat(data.ToArray(), value, data.Dimensions.ToArray());
    }

    public static DenseTensor<Float16> MultipleTensorByFloat(Tensor<Float16> data, Float16 value)
    {
        return MultipleTensorByFloat(data.ToArray(), value, data.Dimensions.ToArray());
    }

    public static DenseTensor<float> AddTensors(float[] sample, float[] sumTensor, int[] dimensions)
    {
        for (var i = 0; i < sample.Length; i++)
        {
            sample[i] = sample[i] + sumTensor[i];
        }
        return CreateTensor(sample, dimensions);
    }

    public static DenseTensor<Float16> AddTensors(Float16[] sample, Float16[] sumTensor, int[] dimensions)
    {
        for (var i = 0; i < sample.Length; i++)
        {
            sample[i] = sample[i].Add(sumTensor[i]);
        }
        return CreateTensor(sample, dimensions);
    }

    public static DenseTensor<float> AddTensors(Tensor<float> sample, Tensor<float> sumTensor)
    {
        return AddTensors(sample.ToArray(), sumTensor.ToArray(), sample.Dimensions.ToArray());
    }

    public static DenseTensor<Float16> AddTensors(Tensor<Float16> sample, Tensor<Float16> sumTensor)
    {
        return AddTensors(sample.ToArray(), sumTensor.ToArray(), sample.Dimensions.ToArray());
    }

    public static Tuple<Tensor<TTensorType>, Tensor<TTensorType>> SplitTensor<TTensorType>(Tensor<TTensorType> tensorToSplit, int[] dimensions)
    {
        var tensor1 = new DenseTensor<TTensorType>(dimensions);
        var tensor2 = new DenseTensor<TTensorType>(dimensions);

        for (int i = 0; i < dimensions[0]; i++)
        {
            for (int j = 0; j < dimensions[1]; j++)
            {
                for (int k = 0; k < dimensions[2]; k++)
                {
                    for (int l = 0; l < dimensions[3]; l++)
                    {
                        tensor1[i, j, k, l] = tensorToSplit[i, j, k, l];
                        tensor2[i, j, k, l] = tensorToSplit[i, j + dimensions[1], k, l];
                    }
                }
            }
        }
        return new Tuple<Tensor<TTensorType>, Tensor<TTensorType>>(tensor1, tensor2);
    }

    public static Tuple<Tensor<TTensorType>, Tensor<TTensorType>> SplitTensor<TTensorType>(Tensor<TTensorType> tensorToSplit, int sections)
    {
        var slicedStrides = tensorToSplit.Dimensions[0] / sections;
        var newDimensions = new int[] {
            tensorToSplit.Dimensions[0] / sections,
            tensorToSplit.Dimensions[1],
            tensorToSplit.Dimensions[2],
            tensorToSplit.Dimensions[3]
        };
        var tensor1 = new DenseTensor<TTensorType>((tensorToSplit as DenseTensor<TTensorType>)!.Buffer.Slice(0 * slicedStrides * tensorToSplit.Strides[0], slicedStrides * tensorToSplit.Strides[0]), newDimensions);
        var tensor2 = new DenseTensor<TTensorType>((tensorToSplit as DenseTensor<TTensorType>)!.Buffer.Slice(1 * slicedStrides * tensorToSplit.Strides[0], slicedStrides * tensorToSplit.Strides[0]), newDimensions);

        return new Tuple<Tensor<TTensorType>, Tensor<TTensorType>>(tensor1, tensor2);
    }

    public static DenseTensor<float> SumTensors(Tensor<float>[] tensorArray, int[] dimensions)
    {
        var sumTensor = new DenseTensor<float>(dimensions);
        var sumArray = new float[sumTensor.Length];

        for (int m = 0; m < tensorArray.Count(); m++)
        {
            var tensorToSum = tensorArray[m].ToArray();
            for (var i = 0; i < tensorToSum.Length; i++)
            {
                sumArray[i] += tensorToSum[i];
            }
        }

        return CreateTensor(sumArray, dimensions);
    }

    public static DenseTensor<Float16> SumTensors(Tensor<Float16>[] tensorArray, int[] dimensions)
    {
        var sumTensor = new DenseTensor<Float16>(dimensions);
        var sumArray = new Float16[sumTensor.Length];

        for (int m = 0; m < tensorArray.Count(); m++)
        {
            var tensorToSum = tensorArray[m].ToArray();
            for (var i = 0; i < tensorToSum.Length; i++)
            {
                sumArray[i] = sumArray[i].Add(tensorToSum[i]);
            }
        }

        return CreateTensor(sumArray, dimensions);
    }

    public static DenseTensor<TTensorType> Duplicate<TTensorType>(TTensorType[] data, int[] dimensions)
    {
        data = data.Concat(data).ToArray();
        return CreateTensor(data, dimensions);
    }

    public static DenseTensor<TTensorType> Concatenate<TTensorType>(TTensorType[] tensor1, TTensorType[] tensor2, int[] dimensions)
    {
        tensor1 = tensor1.Concat(tensor2).ToArray();
        return CreateTensor(tensor1, dimensions);
    }

    public static DenseTensor<float> SubtractTensors(float[] sample, float[] subTensor, int[] dimensions)
    {
        for (var i = 0; i < sample.Length; i++)
        {
            sample[i] = sample[i] - subTensor[i];
        }
        return CreateTensor(sample, dimensions);
    }

    public static DenseTensor<Float16> SubtractTensors(Float16[] sample, Float16[] subTensor, int[] dimensions)
    {
        for (var i = 0; i < sample.Length; i++)
        {
            sample[i] = sample[i].Subtract(subTensor[i]);
        }
        return CreateTensor(sample, dimensions);
    }

    public static DenseTensor<float> SubtractTensors(Tensor<float> sample, Tensor<float> subTensor)
    {
        return SubtractTensors(sample.ToArray(), subTensor.ToArray(), sample.Dimensions.ToArray());
    }

    //public static Tensor<Float16> GetRandomTensorFloat16(ReadOnlySpan<int> dimensions)
    //{
    //    var random = new Random();
    //    var latents = new DenseTensor<Float16>(dimensions);
    //    var latentsArray = latents.ToArray();

    //    for (int i = 0; i < latentsArray.Length; i++)
    //    {
    //        // Generate a random number from a normal distribution with mean 0 and variance 1
    //        var u1 = random.NextDouble(); // Uniform(0,1) random number
    //        var u2 = random.NextDouble(); // Uniform(0,1) random number
    //        var radius = Math.Sqrt(-2.0 * Math.Log(u1)); // Radius of polar coordinates
    //        var theta = 2.0 * Math.PI * u2; // Angle of polar coordinates
    //        var standardNormalRand = radius * Math.Cos(theta); // Standard normal random number
    //        latentsArray[i] = BitConverter.HalfToUInt16Bits((Half)standardNormalRand);
    //    }

    //    latents = CreateTensor(latentsArray, latents.Dimensions.ToArray());

    //    return latents;
    //}

    //public static Tensor<float> GetRandomTensorFloat(ReadOnlySpan<int> dimensions)
    //{
    //    var random = new Random();
    //    var latents = new DenseTensor<float>(dimensions);
    //    var latentsArray = latents.ToArray();

    //    for (int i = 0; i < latentsArray.Length; i++)
    //    {
    //        // Generate a random number from a normal distribution with mean 0 and variance 1
    //        var u1 = random.NextDouble(); // Uniform(0,1) random number
    //        var u2 = random.NextDouble(); // Uniform(0,1) random number
    //        var radius = Math.Sqrt(-2.0 * Math.Log(u1)); // Radius of polar coordinates
    //        var theta = 2.0 * Math.PI * u2; // Angle of polar coordinates
    //        var standardNormalRand = radius * Math.Cos(theta); // Standard normal random number
    //        latentsArray[i] = (float)standardNormalRand;
    //    }

    //    latents = CreateTensor(latentsArray, latents.Dimensions.ToArray());

    //    return latents;
    //}

    public static DenseTensor<TTensorType> Repeat<TTensorType>(DenseTensor<TTensorType> data, int repeats)
    {
        var repeatedData = new TTensorType[data.Length * repeats];
        var newDimensions = new int[] { data.Dimensions[0] * repeats, data.Dimensions[1], data.Dimensions[2] };
        for (int i = 0; i < repeats; i++)
        {
            data.Buffer.CopyTo(repeatedData.AsMemory(i * (int)data.Length));
        }

        return CreateTensor(repeatedData, newDimensions);
    }
}
