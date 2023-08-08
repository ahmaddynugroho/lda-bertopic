def tokenize(df):
    return [str(s).split(' ') for s in df]


def remove_empty_row(df, column_name):
    return df[df[column_name] != '']
