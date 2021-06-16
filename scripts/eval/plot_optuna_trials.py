import joblib
import seaborn as sns
import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--study_loc", type=str)
args = parser.parse_args()

data = joblib.load(args.study_loc)
outloc = args.study_loc.replace(".pkl", ".png")

df = data.trials_dataframe()
df.dropna(inplace=True)
df.reset_index(inplace=True)

df['time'] = df.datetime_complete - df.datetime_start
df['time'] = df.time.astype('int') / 1000000000
df = df[df.time > 0]

names = []

for col in df.columns.values:
    if col[1] == '':
        names.append(col[0])
    else:
        names.append(col[1])

df.columns = names

print('best val:', - round(df.value.min(), 4))
a = sns.lineplot(x=df.index, y=-df.value.cummin())
a.set_xlabel('trial number')
sns.scatterplot(x=df.index, y=-df.value, color='red')
a.set_ylabel('f1 score')
a.legend(['best value', "trial's value"])
plt.savefig(outloc)
