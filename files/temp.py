import fastwer
from torchmetrics.functional import char_error_rate

# Define reference text and output text
ref = 'name'
output = 'nim'

# Obtain Sentence-Level Character Error Rate (CER)
cer = fastwer.score_sent(output, ref, char_level=True)
print(cer)

cer_2 = char_error_rate([output], [ref]).item()
print(cer_2)