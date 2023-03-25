"""To cleanse and visualize the data in the survey results"""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import MaxNLocator

def get_distinct_activity(df):
    activity_list = df.activity.tolist()

    activity_distinct_list = []
    for activity in activity_list:
        str_array = activity.split(';')
        str_array.remove('')
        activity_distinct_list.extend(str_array)
    return sorted(set(activity_distinct_list))


def plot_dept_composition(df):
    fig, ax = plt.subplots(figsize=(13.33,7.5), dpi=96)
    # Plot the bars
    dept_cnt = df.groupby('department')['department'].count()
    # print(dept_cnt.index)
    # print(dept_cnt.values)
    bars = ax.bar(x=dept_cnt.index, height=dept_cnt.values)
    ax.bar_label(bars)

    # This works but can't add label ontop of the bars
    # bars = df['department'].value_counts()[:5].plot(kind='bar')

    # adding the grids or rulers
    ax.grid(which='major', axis='y', color='#DAD8D7', alpha=0.5, zorder=1)
    ax.grid(which='major', axis='x', color='#DAD8D7', alpha=0.5, zorder=1)

    # Reformat x-axis label and tick labels
    ax.set_xlabel('Departments', fontsize=12, labelpad=10)
    ax.xaxis.set_label_position('bottom')
    ax.xaxis.set_tick_params(pad=2, labelbottom=True, bottom=True, labelsize=12, labelrotation=0)

    # Reformat y-axis
    ax.set_ylabel('Count', fontsize=12, labelpad=10)
    ax.yaxis.set_label_position('left')
    ax.yaxis.set_major_formatter(lambda s, i : f'{s:,.0f}')
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_tick_params(pad=2, labeltop=False, labelbottom=True, bottom=False, labelsize=12)

    # Title
    ax.text(x=0.3, y=.93, s="Count of survey result from each department", transform=fig.transFigure, ha='left',
        fontsize=14, weight='bold', alpha=.8)
    
    # plt.savefig("/Users/kangwei/development/repo/notebookRepo/surveyResult/img/deptCompo.png")
    plt.show()


def plot_activity_count(activity_list, df):
    # 'Badminton Tournament', 'Batik painting', 'Bowling Tournament', 
    # 'CSR', 'Camping', 'Coral Planting', 'Culinary Classes', 
    # 'Family day outing', 'Fun Run', 'Hackathon', 'Lunch/Dinner Gathering', 
    # 'Movie Night', 'Outdoor Activity e.g. hiking, beach cleanup', 
    # 'PS5 games', 'Paintball', 'Soap Making Classes', 'Sports Day'
    activity_count = []
    for activity in activity_list:
        cnt = df[activity].sum()
        activity_count.append(cnt)

    # Shorten some labels before passing into the plots
    activity_list = list(map(lambda x: x.replace('Outdoor Activity e.g. hiking, beach cleanup', 'Outdoor'), activity_list))
    activity_list = list(map(lambda x: x.replace('Family day outing', 'Family day'), activity_list))
    activity_list = list(map(lambda x: x.replace('Badminton Tournament', 'Badminton'), activity_list))
    activity_list = list(map(lambda x: x.replace('Batik painting', 'Batik'), activity_list))
    activity_list = list(map(lambda x: x.replace('Bowling Tournament', 'Bowling'), activity_list))
    activity_list = list(map(lambda x: x.replace('Lunch/Dinner Gathering', 'Meal'), activity_list))
    activity_list = list(map(lambda x: x.replace('Coral Planting', 'Coral'), activity_list))
    activity_list = list(map(lambda x: x.replace('Soap Making Classes', 'Soap Making'), activity_list))
    activity_list = list(map(lambda x: x.replace('Culinary Classes', 'Culinary'), activity_list))

    fig, ax = plt.subplots(figsize=(13,8), dpi=96)
    bars = ax.bar(x=activity_list, height=activity_count)
    ax.bar_label(bars)

    # adding the grids or rulers
    ax.grid(which='major', axis='y', color='#DAD8D7', alpha=0.5, zorder=1)
    ax.grid(which='major', axis='x', color='#DAD8D7', alpha=0.5, zorder=1)

    # Reformat x-axis label and tick labels
    ax.set_xlabel('Activities', fontsize=12, labelpad=10)
    # ax.xaxis.set_label_position('bottom')
    ax.xaxis.set_tick_params(pad=2, labelbottom=True, bottom=True, labelsize=11, labelrotation=60)

    # Reformat y-axis
    ax.set_ylabel('Count', fontsize=12, labelpad=10)
    ax.yaxis.set_label_position('left')
    ax.yaxis.set_major_formatter(lambda s, i : f'{s:,.0f}')
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_tick_params(pad=2, labeltop=False, labelbottom=True, bottom=False, labelsize=12)

    # Title
    ax.text(x=0.3, y=.93, s="Suggested Activities", transform=fig.transFigure, ha='left',
        fontsize=14, weight='bold', alpha=.8)
    
    plt.savefig("/Users/kangwei/development/repo/notebookRepo/surveyResult/img/activityCount.png")
    plt.show()



def main():
    df = pd.read_csv('/Users/kangwei/development/repo/notebookRepo/surveyResult/survey.csv', header=0)

    # 1. get a distinct list of the activities in the survey result
    # 2. make the activities into columns
    # 3. visualization 1: composition of participants
    # 4. visualization 2: activity count
    activity_list = get_distinct_activity(df)
    # print(activity_list)
    for activity in activity_list:
        df[activity] = df['activity'].str.contains(activity)
    
    # plot_dept_composition(df)
    plot_activity_count(activity_list, df)



if __name__ == '__main__':
    main()





