﻿// Copyright (C) 2023 Gianni Rosa Gallina. All rights reserved.
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
using System.Runtime.CompilerServices;

public static class Float16Extensions
{
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Float16 Add(this Float16 t1, Float16 t2)
    {
        var half1 = (float)t1;
        var half2 = (float)t2;

        return (Float16)(half1 + half2);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Float16 Subtract(this Float16 t1, Float16 t2)
    {
        var half1 = (float)t1;
        var half2 = (float)t2;
        
        return (Float16)(half1 - half2);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Float16 Mul(this Float16 t1, Float16 t2)
    {
        var half1 = (float)t1;
        var half2 = (float)t2;

        return (Float16)(half1 * half2);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Float16 Div(this Float16 t1, Float16 t2)
    {
        var half1 = (float)t1;
        var half2 = (float)t2;

        return (Float16)(half1 / half2);
    }
}
