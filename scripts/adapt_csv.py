import csv
import statistics

with open('data/annotations/arousal_cont_average.csv', "r") as cont_arousal_avg: 
    reader = csv.reader(cont_arousal_avg)
    with open('data/annotations_new/arousal_cont_10.csv', 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)

        for row in reader: 
            song_id = str(row[0])
            if song_id != 'song_id':
                firstclip = [float(item) for item in row[1:22]]
                secondclip = [float(item) for item in row[21:42]]
                thirdclip = [float(item) for item in row[41:62]]
                
                average1 = statistics.mean(firstclip)
                std_deviation1 = statistics.stdev(firstclip)
                
                average2 = statistics.mean(secondclip)
                std_deviation2 = statistics.stdev(secondclip)

                average3 = statistics.mean(thirdclip)
                std_deviation3 = statistics.stdev(thirdclip)

                csv_writer.writerow([song_id+"_1", average1, std_deviation1])
                csv_writer.writerow([song_id+"_2", average2, std_deviation2])
                csv_writer.writerow([song_id+"_3", average3, std_deviation3])

with open('data/annotations/valence_cont_average.csv', "r") as cont_valence_avg: 
    reader = csv.reader(cont_valence_avg)
    with open('data/annotations_new/valence_cont_10.csv', 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)

        for row in reader: 
            song_id = str(row[0])
            if song_id != 'song_id':
                firstclip = [float(item) for item in row[1:22]]
                secondclip = [float(item) for item in row[21:42]]
                thirdclip = [float(item) for item in row[41:62]]
                
                average1 = statistics.mean(firstclip)
                std_deviation1 = statistics.stdev(firstclip)
                
                average2 = statistics.mean(secondclip)
                std_deviation2 = statistics.stdev(secondclip)

                average3 = statistics.mean(thirdclip)
                std_deviation3 = statistics.stdev(thirdclip)

                csv_writer.writerow([song_id+"_1", average1, std_deviation1])
                csv_writer.writerow([song_id+"_2", average2, std_deviation2])
                csv_writer.writerow([song_id+"_3", average3, std_deviation3])

