# Segment Any Text CoreML
Segment Any Text is state-of-the-art sentence segmentation with 3 Transfomer layers. A pytorch version of the model is used in [wtsplit](https://github.com/segment-any-text/wtpsplit) and additional details can be found in this [paper](https://arxiv.org/abs/2406.16678).

If you wish to skip the CoreML conversion, you can download a precompiled `SaT.mlmodelc` from [Hugging Face](https://huggingface.co/smdesai/SaT).

## Swift Usage

Swift sample code for testing and integrating the Core ML model is available at [smdesai/SegmentText](https://github.com/smdesai/SegmentText).


# CoreML Conversion

## Environment Setup

1. Install [uv](https://github.com/astral-sh/uv) if it is not already available.
2. Sync the project environment.
   ```bash
   uv sync
   ```
3. Activate the virtual environment:
   ```bash
   source .venv/bin/activate
   ```

## Converting the Model

Run the conversion script to create the SaT Core ML package:

```bash
python convert-sat.py --model-id segment-any-text/sat-3l-sm  --output-dir sat_coreml
```

This produces `SaT.mlpackage` in the `sat_coreml` directory.

Here is the complete usage:
```bash
Usage: convert_sat.py [OPTIONS]

 Options
  --model-id                 TEXT  Model identifier to download      
                                   from HuggingFace model hub      
                                   [default:                         
                                   segment-any-text/sat-3l-sm]       
  --output-dir               PATH  Directory to write mlpackage and  
                                   [default: sat_coreml]             
  --conversion-type  -c      TEXT  Conversion methods to apply to    
                                   the model. Repeat the option to   
                                   chain conversions (allowed:       
                                   none, prune, quantize,            
                                   palettize; default: none).        
                                   [default: None]                   
```

## Compiling the Model

Run the following to compile the model.
```bash
python compile_mlmodelc.py --coreml-dir sat_coreml --output-dir compiled
```

This produces `SaT.mlmodelc` in the `compiled` directory.

Here is the complete usage:
```bash
 Usage: compile_mlmodelc.py [OPTIONS]

 Options
  --coreml-dir        PATH  Directory where the mlpackage is
                            [default: sat_coreml]
  --output-dir        PATH  Directory where the compiled model is written
                            [default: compiled]
```
