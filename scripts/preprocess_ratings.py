import csv

if __name__ == '__main__':
    input_file = './output_plays_small.csv'
    output_file = './output_plays_small_preprocessed.csv'
    with open(output_file, mode='w', encoding='UTF8', newline='') as write_f:
        writer = csv.writer(write_f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

        with open(input_file) as read_f:
            csv_reader = csv.DictReader(read_f)
            line_count = 0
            for row in csv_reader:
                if line_count == 0:
                    print(f'Column names are {", ".join(row)}')
                    writer.writerow(['userID', 'songID', 'playCount'])
                    line_count += 1
                else:
                    line_count += 1
                    c = int(row['playCount'])
                    new_value = 5
                    if c <= 50:
                        new_value = 4
                    if c <= 40:
                        new_value = 3
                    if c <= 30:
                        new_value = 2
                    if c <= 20:
                        new_value = 1
                    if c <= 10:
                        new_value = 0
                    writer.writerow([row["userID"], row["songID"], new_value])

            print(f'Processed {line_count} lines.')
