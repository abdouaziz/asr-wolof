import tensorflow as tf 
 
def compute_ctc_loss (logits , labels , logit_length , label_length):
    """
    Compute CTC loss

    Args:
        logits: Tensor of shape [batch_size, max_time_steps, num_classes]
        labels: Tensor of shape [batch_size, max_time_steps]
        logit_length: Tensor of shape [batch_size]
        label_length: Tensor of shape [batch_size]

    Returns:
        loss: Tensor of shape [batch_size]
    """
  
    return tf.nn.ctc_loss (labels=labels, 
                         logits=logits, 
                         label_length=label_length,
                         logit_length=logit_length,
                         logits_time_major=False,
                         unique=None,
                         blank_index=1,
                         name=None)