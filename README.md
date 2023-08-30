watermarking
==============================

analysis of watermarkig LMs

watermark_processor = WatermarkLogitsProcessor(vocab=list(tokenizer.get_vocab().values()),
                                               gamma=0.25,
                                               delta=2.0,
                                               seeding_scheme="simple_1")

tokenized_input = tokenizer(input_text).to(model.device)
# note that if the model is on cuda, then the input is on cuda
# and thus the watermarking rng is cuda-based.
# This is a different generator than the cpu-based rng in pytorch!

output_tokens = model.generate(**tokenized_input,
                               logits_processor=LogitsProcessorList([watermark_processor]))

# if decoder only model, then we need to isolate the
# newly generated tokens as only those are watermarked, the input/prompt is not
output_tokens = output_tokens[:,tokenized_input["input_ids"].shape[-1]:]

output_text = tokenizer.batch_decode(output_without_watermark, skip_special_tokens=True)[0]
```

Detect watermarked text:

```python
watermark_detector = WatermarkDetector(vocab=list(tokenizer.get_vocab().values()),
                                        gamma=0.25, # should match original setting
                                        seeding_scheme="simple_1", # should match original setting
                                        device=model.device, # must match the original rng device type
                                        tokenizer=tokenizer,
                                        z_threshold=4.0,
                                        normalizers=[],
                                        ignore_repeated_bigrams=False)

score_dict = watermark_detector.detect(output_text) # or any other text of interest to analyze
```

