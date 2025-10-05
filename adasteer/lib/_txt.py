

def load_txt(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        lines = [line.strip() for line in file.readlines()]  # strip() 去掉每行的换行符
    return lines