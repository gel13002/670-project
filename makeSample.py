"""
    take first 2000 samples from data.csv
"""
import csv

with open('data.csv', 'r') as csvfile:
    with open('sampledata.csv', 'w') as sample:
        reader = csv.reader(csvfile)
        writer = csv.writer(sample)
        rowCount = 0
        expr_table = dict()
        for row in reader:
            expression = row[-1]
            expr_table[expression] = expr_table.get(expression, 0) + 1
            rowCount += 1
            writer.writerow(row)
            if rowCount > 2000:
                break

for expr, count in expr_table.items():
    print(expr, count)
