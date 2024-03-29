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

using SharpDiffusion.Interfaces;

namespace SharpDiffusion.Schedulers;

public class SchedulerFactory
{
    public IScheduler GetScheduler<TTensorType>(SchedulerType schedulerType)
    {
        return schedulerType switch
        {
            SchedulerType.LMSDiscreteScheduler => typeof(TTensorType) == typeof(float) ? new LMSDiscreteScheduler() : new LMSDiscreteSchedulerFloat16(),
            //SchedulerType.EulerAncestralDiscreteScheduler => new EulerAncestralDiscreteScheduler<TTensorType>(),
            _ => throw new InvalidOperationException($"Unsupported scheduler type '{schedulerType}'"),
        };
    }
}