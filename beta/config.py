import csv 

#Replace with the path of your config file; refer to documentation for more information
config_path = ''

if config_path == '':config_path = '/home/bdyee/config/config.csv'
with open(config_path, 'r') as f:
            reader = csv.reader(f)
            config = []
            for row in reader:
                config.append(row)