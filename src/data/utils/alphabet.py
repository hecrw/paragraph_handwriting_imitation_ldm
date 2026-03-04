import string
import torch

from src.data.utils.constants import *


def convert_position_to_text(pos, d):
    out = []
    for p in pos:
        out.append(d[p])
    out = "".join(out)
    return out

class Alphabet():
    def __init__(self, dataset="UNIFIED", mode="both"):
        if mode == "ctc":
            extra = [BLANK,PAD]
        elif mode== "attention":
            extra = [BLANK,PAD,START_OF_SEQUENCE,END_OF_SEQUENCE,NEW_LINE]
        elif mode== "both":
            extra = [BLANK, PAD, START_OF_SEQUENCE, END_OF_SEQUENCE,NEW_LINE]
        self.seperators = None
        lower = string.ascii_lowercase
        upper = string.ascii_uppercase
        numbers = string.digits
        punctuation = [' ', '.', ',',"'",'-','"','#','(',')',':',';','?','*','!','/','&','+']
        arrows = [u"\u2190", u"\u2191", u"\u2192", u"\u2193", u"\u2194"]
        nbb_characters = ['ſ', 'ʒ', 'Ʒ', 'Ü', 'ü', 'Ö', 'ö', 'Ä', 'ä', ]  # according to transcription guidelines
        rimes_characters = ["à", "é", "è", "€", "ù", "ô", 'ê', 'ç', 'î', 'û', '¤', '°', 'â', 'ë', '=', '{', '}', 'À',
                            '%', 'É', '²', 'œ', '_', ]
        arabic_letters = [' ', 'ء', 'ا', 'ب', 'ة', 'ت', 'ث', 'ج', 'ح', 'خ',
                          'د', 'ذ', 'ر', 'ز', 'س', 'ش', 'ص', 'ض', 'ط', 'ظ',
                          'ع', 'غ', 'ف', 'ق', 'ك', 'ل', 'م', 'ن', 'ه', 'و', 'ي']

        if dataset.lower() == "unified" or dataset.lower() == "arabic":
            types = [extra, arabic_letters]
        elif dataset.lower()=="NBB".lower():
            self.seperators = ['.',':',',','/','~','§','%','-','*',' ','@']
            types = [extra, lower, upper, numbers, nbb_characters, self.seperators]
        elif dataset=="rimes":
            types = [extra, lower, upper, numbers, punctuation, rimes_characters]
        elif dataset=="all":
            types = [extra, lower, upper, numbers, punctuation, arrows]
        else: #IAM case
            types = [extra,lower,upper,numbers, punctuation]

        self.toPosition = {}
        self.toCharacter = {}
        self.labels = []
        id = 0
        for t in types:
            for char in t:
                self.toPosition[char] = id
                self.toCharacter[id] = char
                id += 1
                self.labels.append(char)

    def string_to_logits(self, x_in):
        out = []
        for i in x_in:
            out.append(self.toPosition[i])
        return torch.LongTensor(out)

    def logits_to_string(self, x_in):
        out = []
        for i in x_in:
            out.append(self.toCharacter[int(i)])
        return "".join(out)

    def batch_logits_to_string_list(self, x_in, stopping_logits: list = None):
        text = []
        classification = []
        for b in x_in:
            if stopping_logits is None:
                text.append(self.logits_to_string(b))
                classification.append(torch.Tensor([self.toPosition[PAD]]))
            else:
                stops = []
                for s in stopping_logits:
                    stop = torch.where(b == s)[0]
                    if len(stop) == 0:
                        stop = torch.LongTensor([len(b)])
                    stops.append(stop[0])
                end_idx = torch.min(torch.stack(stops))
                text.append(self.logits_to_string(b[:end_idx]))
                if end_idx == len(b):
                    classification.append(torch.Tensor([self.toPosition[PAD]]))
                else:
                    end_classifier = torch.argmin(torch.stack(stops))
                    classification.append(torch.Tensor([stopping_logits[end_classifier]]))
        return text, torch.stack(classification)

    # def batch_logits_to_string_list(self, x_in):
    #     out = []
    #     for b in x_in:
    #         out.append(self.logits_to_string(b))
    #     return out

if __name__ == "__main__":
    A = Alphabet(dataset="iam", mode="both")
    print(A.labels)
    logits = A.string_to_logits("bla")
    print(A.logits_to_string(logits))
    batch = torch.LongTensor([[1,2,3,4],[2,2,2,2]])
    print(batch.shape)
    print(A.batch_logits_to_string_list(batch))

