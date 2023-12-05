# -*- coding: utf-8 -*-
import os


class FileHandler:

    def __init__(self, path,mode):
        self._path = path
        self.file = open(path, mode, encoding='utf-8')

    def write(self, line):
        """
        msg: '"abc,vvd","123,cv"'(csv文件写,)
        """
        self.file.write(line + '\n')
        self.file.flush()

    def close(self):
        self.file.close()


class RealTimeCsv(FileHandler):
    """csv文件实时写入"""

    def __init__(self, path,mode):
        super().__init__(path, mode)
        self.columns = []

    def dict_write_row(self, row_data: dict):
        """
        row_data: dict
        """
        row_msg = ''
        for col in self.columns:
            row_msg += '{},'.format(row_data.get(col, ''))
        row_msg = row_msg.strip(',')
        self.write(row_msg)

    def list_write_row(self, row_data: list):
        """
        row_data: list
        """
        self.write(str(row_data)[1:-1].replace("\n", "").replace("\r", ""))

    def set_columns(self, columns: list):
        """设置csv文件头"""
        self.columns = columns
        row_msg = ''
        for item in columns:
            row_msg += '{},'.format(item)
        row_msg = row_msg.strip(',')
        self.write(row_msg)

if __name__ == '__main__':
    # demo
    # file = RealTimeCsv('test.csv','a+')
    # file.set_columns(["auc", "y_true", "y_pred"])
    # for i in range(20):
    #     file.list_write_row([1,[2,3],[1,2]])
    # file.close()
    import pandas as pd
    myList = [[['auc', 'y_pred', 'y_true', 'sample_num', 'epoch', 'train_loss'], [0.4977231672437152, 0.5388866956178444, 0.0, 0.0, 0.0, 2, 0, 0]]]
    df1 = pd.DataFrame(data=myList)
    pd.DataFrame(myList).to_csv('test.csv',index=False,header=False)
    print(df1)