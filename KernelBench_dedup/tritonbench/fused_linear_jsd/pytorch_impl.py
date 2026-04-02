import torch
import torch.nn as nn
import torch.nn.functional as F


class TorchJSD(nn.Module):
    """Jensen-Shannon Divergence loss."""

    def __init__(self, beta=0.5, ignore_index=-100):
        super().__init__()
        self.kl = nn.KLDivLoss(reduction="none", log_target=True)
        self.beta = beta
        self.ignore_index = ignore_index

    def forward(self, log_q, log_p, label=None):
        log_p, log_q = log_p.to(torch.float), log_q.to(torch.float)
        log_p = log_p.view(-1, log_p.size(-1))
        log_q = log_q.view(-1, log_q.size(-1))
        m = torch.lerp(torch.exp(log_q), torch.exp(log_p), self.beta)
        loss = self.beta * self.kl(torch.log(m), log_p).sum(dim=-1) + (
            1 - self.beta
        ) * self.kl(torch.log(m), log_q).sum(dim=-1)

        if label is not None:
            loss = torch.where(label != self.ignore_index, loss, 0.0)
            n_non_ignore = (label != self.ignore_index).sum().item()
            if n_non_ignore == 0:
                loss = 0.0
            else:
                loss = (loss / n_non_ignore).sum()
        else:
            loss = (loss / log_q.shape[0]).sum()
        return loss


class Model(nn.Module):
    """Fused linear + JSD baseline: linear projections then JSD loss.

    The real operator fuses two linear projections (student + teacher) with
    JSD loss. This adapter performs them sequentially.
    """

    def __init__(self, hidden_size=4096, vocab_size=128256, dtype=torch.float32,
                 beta=0.5, temperature=1.0):
        super().__init__()
        self.student_lin = nn.Linear(hidden_size, vocab_size, bias=False)
        self.teacher_lin = nn.Linear(hidden_size, vocab_size, bias=False)
        self.jsd = TorchJSD(beta=beta)
        self.temperature = temperature

    def forward(self, student_input, teacher_input):
        student_logits = self.student_lin(student_input)
        teacher_logits = self.teacher_lin(teacher_input)
        student_prob = torch.log_softmax(student_logits / self.temperature, dim=-1)
        teacher_prob = torch.log_softmax(teacher_logits / self.temperature, dim=-1)
        return self.jsd(student_prob, teacher_prob)


# Default shapes
HIDDEN_SIZE = 4096
VOCAB_SIZE = 128256
BT = 1024  # batch*time dimension (2^10)
DTYPE = torch.float32


def get_inputs():
    student_input = torch.randn(BT, HIDDEN_SIZE, dtype=DTYPE)
    teacher_input = torch.randn(BT, HIDDEN_SIZE, dtype=DTYPE)
    return [student_input, teacher_input]


def get_init_inputs():
    return [HIDDEN_SIZE, VOCAB_SIZE, DTYPE]


def get_test_inputs():
    return [x.cuda() if isinstance(x, torch.Tensor) else x for x in get_inputs()]


def run(*args):
    if args:
        inputs = list(args)
    else:
        inputs = get_test_inputs()
    model = Model(*get_init_inputs()).cuda().eval()
    with torch.no_grad():
        return model(*inputs)
