# Third-party notices

## NeuroKit / Kubios-style artifact classification

`offline_processing/gp_hrv_artifacts.py` contains the interval-based adaptation
of NeuroKit's Lipponen–Tarvainen artifact classifier from the user's GP_pipeline.
The classifier functions are retained unchanged from that local source. This
project uses artifact indices only and does not invoke its peak-correction code.

Original project: https://github.com/neuropsychology/NeuroKit
Reference: Lipponen & Tarvainen (2019), https://doi.org/10.1080/03091902.2019.1640306

The NeuroKit distribution supplies the following license:

MIT License

Copyright (c) 2020, Dominique Makowski

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
