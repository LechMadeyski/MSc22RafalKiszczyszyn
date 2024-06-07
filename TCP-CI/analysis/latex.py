import pandas as pd

def encode(avg, std):
    return "$" + "{:.2f}".format(avg) + "\\pm" + "{:.2f}".format(std) + "$"

def decode(value: str):
    return list(map(lambda f: float(f), value.split(" ± ")))

def to_latex_row(sid: str, row: dict, columns: list):
    id = int(sid[1:])
    transformed = []
    for column in columns:
        cell = row[column]
        if "score" in cell:
            score = cell["score"]
            score = encode(score["avg"], score["std"])
            score = f"\\bm{{{score}}}" if cell["bold"] else score
            transformed.append(score)
        elif "p_value" in cell:
            p_value = cell["p_value"]
            if p_value < 0.01:
                p_value = "\\textbf{<0.01}"
            elif p_value < 0.05:
                p_value = f"\\textbf{{{p_value}}}"
            transformed.append(p_value)
        else:
            transformed.append(cell["text"])
            
    return f"$S_{{{id}}}$ & " + " & ".join(transformed) + "\\\\"

def find_best(scores: list):
    index_ = 0
    max_avg, min_std = 0, 1
    for index, score in enumerate(scores):
        avg, std = score
        if avg > max_avg or (avg == max_avg and std < min_std):
            max_avg = avg
            min_std = std
            index_ = index
    
    return index_


def convert(score):
    return {"avg": score[0], "std": score[1]}


WORKDIR = 'C:\\Users\\rafal\\MT\\repos\\MSc22RafalKiszczyszyn\\TCP-CI\\analysis\\'
df = pd.read_csv(WORKDIR + 'experiment0.csv')
for _, row in df.iterrows():
    scores = [decode(score) for score in row[["FULL APFD", "FULL APFD", "H APFD", "RND APFD"]].to_list()]
    index = find_best(scores)

    row_ = {
        0: {"score": convert(scores[0]), "bold": index == 0},
        1: {"score": convert(scores[1]), "bold": index == 1},
        2: {"score": convert(scores[2]), "bold": index == 2},
        3: {"score": convert(scores[3]), "bold": index == 3},
    }

    lrow = to_latex_row(row['SID'], row_, columns=[0, 1, 2, 3])
    print(lrow)
