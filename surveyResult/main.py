"""To cleanse and visualize the data in the survey results"""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from matplotlib.ticker import MaxNLocator

def get_distinct_activity(df):
    activity_list = df.activity.tolist()

    activity_distinct_list = []
    for activity in activity_list:
        str_array = activity.split(';')
        str_array.remove('')
        activity_distinct_list.extend(str_array)
    
    # Shorten some labels before passing into the plots
    activity_distinct_list = list(map(lambda x: x.replace('Outdoor Activity e.g. hiking, beach cleanup', 'Outdoor'), activity_distinct_list))
    activity_distinct_list = list(map(lambda x: x.replace('Family day outing', 'Family day'), activity_distinct_list))
    activity_distinct_list = list(map(lambda x: x.replace('Badminton Tournament', 'Badminton'), activity_distinct_list))
    activity_distinct_list = list(map(lambda x: x.replace('Batik painting', 'Batik'), activity_distinct_list))
    activity_distinct_list = list(map(lambda x: x.replace('Bowling Tournament', 'Bowling'), activity_distinct_list))
    activity_distinct_list = list(map(lambda x: x.replace('Lunch/Dinner Gathering', 'Lunch'), activity_distinct_list))
    activity_distinct_list = list(map(lambda x: x.replace('Coral Planting', 'Coral'), activity_distinct_list))
    activity_distinct_list = list(map(lambda x: x.replace('Soap Making Classes', 'Soap Making'), activity_distinct_list))
    activity_distinct_list = list(map(lambda x: x.replace('Culinary Classes', 'Culinary'), activity_distinct_list))
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
    # 'Badminton', 'Batik', 'Bowling', 'CSR', 'Camping', 'Coral', 'Culinary', 
    # 'Family day', 'Fun Run', 'Hackathon', 'Lunch', 'Movie Night', 'Outdoor', 
    # 'PS5 games', 'Paintball', 'Soap Making', 'Sports Day'
    activities = df[['Badminton', 'Batik', 'Bowling', 'CSR', 'Camping', 'Coral', 'Culinary', 
                     'Family day', 'Fun Run', 'Hackathon', 'Lunch', 'Movie Night', 'Outdoor', 
                     'PS5 games', 'Paintball', 'Soap Making', 'Sports Day']]
    activities_sum = activities.sum()

    fig, ax = plt.subplots(figsize=(13,8), dpi=96)
    bars = ax.bar(x=activities_sum.index, height=activities_sum.values)
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


def plot_dept_activity_count(activity_list, df):
    activities = df[['department', 'Badminton', 'Batik', 'Bowling', 'CSR', 'Camping', 'Coral', 'Culinary', 
                     'Family day', 'Fun Run', 'Hackathon', 'Lunch', 'Movie Night', 'Outdoor', 
                     'PS5 games', 'Paintball', 'Soap Making', 'Sports Day']]
    activities_grouped = activities.groupby('department').sum()
    # print(activities_grouped.head())

    # Melt makes the columns into a value in rows
    activities_grouped = activities_grouped.reset_index().melt(id_vars=['department'], var_name='activity', value_name='count')
    # print(activities_grouped.head())

    # fig, axes = plt.subplots(figsize=(13,8), dpi=96, sharey='col')
    # Can be used to change the figure size of sns.scatterplot and sns.boxplot
    # sns.set(rc={"figure.figsize":(12, 20)})
    sns.set_style("whitegrid")

    # Create the count plot
    #height=8 width=1.5 times the height
    activity_plt = sns.catplot(x='activity', y='count', hue='department', data=activities_grouped, kind='bar', 
                               height=7, aspect=1.5, orient='v', legend_out=False, margin_titles=False)\
        .set(title="Suggested activities grouped by Department")\
        .set_xticklabels(rotation=20)
    
    plt.savefig("/Users/kangwei/development/repo/notebookRepo/surveyResult/img/deptActivityCount.png")
    plt.show()


def plot_frequency_count(df):
    frequency_df = df[['department', 'frequency']]
    frequency_list = frequency_df['frequency'].unique()
    # print(frequency_df.head())
    # print(frequency_list)

    sns.countplot(data=frequency_df, x='frequency').set(title='Frequency count')
    plt.savefig("/Users/kangwei/development/repo/notebookRepo/surveyResult/img/frequencyCount.png")
    plt.show()
    sns.countplot(data=frequency_df, x='frequency', hue='department').set(title='Dept Frequency count')    
    plt.savefig("/Users/kangwei/development/repo/notebookRepo/surveyResult/img/deptFrequencyCount.png")
    plt.show()


def plot_weekday_weekend_count(df):
    tmp_df = df[['department', 'weekdays_weekends']]
    unique_list = tmp_df['weekdays_weekends'].unique()
    # print(unique_list)
    tmp_df = tmp_df.replace('Only on weekends', 'weekends')
    tmp_df = tmp_df.replace('Only on weekdays', 'weekdays')
    tmp_df = tmp_df.replace('Either is good, depending on the activities.', 'any')

    sns.countplot(data=tmp_df, x='weekdays_weekends').set(title='Preferred day count')
    plt.savefig("/Users/kangwei/development/repo/notebookRepo/surveyResult/img/DayCount.png")
    plt.show()
    sns.countplot(data=tmp_df, x='weekdays_weekends', hue='department').set(title='Dept Day count')    
    plt.savefig("/Users/kangwei/development/repo/notebookRepo/surveyResult/img/deptDayCount.png")
    plt.show()


def plot_workshop(df):
    tmp_df = df[['department', 'attend_workshop']]
    # unique_list = tmp_df['attend_workshop'].unique()
    # print(unique_list)

    sns.countplot(data=tmp_df, x='attend_workshop').set(title='Workshop preference')
    plt.savefig("/Users/kangwei/development/repo/notebookRepo/surveyResult/img/workshopCount.png")
    plt.show()
    sns.countplot(data=tmp_df, x='attend_workshop', hue='department').set(title='Workshop preference by Dept')    
    plt.savefig("/Users/kangwei/development/repo/notebookRepo/surveyResult/img/deptWorkshopCount.png")
    plt.show()


def main():
    df = pd.read_csv('/Users/kangwei/development/repo/notebookRepo/surveyResult/survey.csv', header=0)

    # 1. get a distinct list of the activities in the survey result
    # 2. make the activities into columns
    # 3. visualization 1: composition of participants
    # 4. visualization 2: activity count
    # 5. visualization 3: activity with department count
    activity_list = get_distinct_activity(df)
    # print(activity_list)
    for activity in activity_list:
        df[activity] = df['activity'].str.contains(activity)
    
    # plot_dept_composition(df)
    # plot_activity_count(activity_list, df)
    # plot_dept_activity_count(activity_list, df)
    # plot_frequency_count(df)
    # plot_weekday_weekend_count(df)
    # plot_workshop(df)



if __name__ == '__main__':
    main()

