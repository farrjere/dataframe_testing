import pandas as pd
import polars as pl
import pyarrow as pa
from memory_profiler import profile
import matplotlib.pyplot as plt
import numpy as np
from pyarrow import csv
import pyarrow.compute as pc

#@profile
def pandas_test():
    # loading and basic stats
    job_data = pd.read_csv('job_postings.csv')
    grouped_wt = job_data.groupby(by='work_type')['med_salary']
    salary_by_work_type = grouped_wt.agg(['min', 'median', 'max']).reset_index()
    print(salary_by_work_type)
    
    # basic plot
    plot_med_salary(salary_by_work_type['median'], salary_by_work_type['work_type'], 'pandas_med_plot.png')
    
    # combining data test
    sales_mgr_jobs = job_data[job_data['title'] == 'Sales Manager']
    software_jobs = job_data[job_data['title'].str.startswith('Software Engineer')]
    frames = []
    for i in range(10):
        frames.append(sales_mgr_jobs)
        frames.append(software_jobs)

    comb_jobs = pd.concat(frames)

    # misc windows and sorts
    job_data = job_data.sort_values('med_salary')
    wt_unique = job_data['work_type'].nunique()
    print("unique job types:", wt_unique)
    jd_quant = job_data.quantile([.1, .5, 0.9], method="table", interpolation="nearest")
    print(jd_quant)

#@profile
def polars_test():
    job_data = pl.read_csv('job_postings.csv')
    grouped_wt = job_data.group_by("work_type").agg(
        pl.min("med_salary").alias('min'),
        pl.median("med_salary").alias('median'),
        pl.max("med_salary").alias('max')
    )
    print(grouped_wt)
    
    plot_med_salary(grouped_wt['median'], grouped_wt['work_type'], 'polars_med_plot.png')
    

    sales_mgr_jobs = job_data.filter(pl.col("title") == 'Sales Manager')
    software_jobs = job_data.filter(pl.col('title').str.starts_with('Software Engineer'))

    combined = sales_mgr_jobs
    for _ in range(10):
        combined.vstack(sales_mgr_jobs, in_place=True)
        combined.vstack(software_jobs, in_place=True)

    job_data = job_data.sort("med_salary")
    wt_unique = job_data.n_unique(subset=['work_type'])
    print("unique job types:", wt_unique)
    jd_quant_1 = job_data.quantile(.1, 'nearest')
    jd_quant_5 = job_data.quantile(.5, 'nearest')
    jd_quant_9 = job_data.quantile(.9, 'nearest')
    print(jd_quant_1, jd_quant_5, jd_quant_9)

@profile
def pyarrow_test():
    job_data = csv.read_csv('job_postings.csv', parse_options=csv.ParseOptions(newlines_in_values=True))
    groupted_wt = pa.TableGroupBy(job_data, "work_type")
    salary_by_work_type = groupted_wt.aggregate([("med_salary", "min"), ("med_salary", "approximate_median"), ("med_salary", "max")])
    print(salary_by_work_type)
    
    # basic plot
    plot_med_salary(salary_by_work_type['med_salary_approximate_median'], salary_by_work_type['work_type'], 'pyarrow_med_plot.png')
    sales_filter = (pc.field("title") == 'Sales Manager')
    software_filter = (pc.starts_with(pc.field('title'), 'Software Engineer'))
    sales_mgr_jobs = job_data.filter(sales_filter)
    software_jobs = job_data.filter(software_filter)

    frames = []
    for i in range(10):
        frames.append(sales_mgr_jobs)
        frames.append(software_jobs)

    comb_jobs = pa.concat_tables(frames)
    job_data = job_data.sort_by('med_salary')
    wt_unique = pc.count_distinct(job_data['work_type'])
    print("unique job types:", wt_unique)
    numeric_fields = [f.name for f in job_data.schema if not f.type.equals(pa.string())]
    jd_quant = [pc.quantile(job_data[f], [.1, .5, 0.9], interpolation="nearest") for f in numeric_fields]
    print(jd_quant)

def plot_med_salary(salary, job_types, file):
    plt.rcdefaults()
    fig, ax = plt.subplots()
    y_pos = np.arange(len(job_types))
    ax.barh(y_pos, salary, align='center')
    ax.set_yticks(y_pos, labels=job_types)
    ax.invert_yaxis()  # labels read top-to-bottom
    ax.set_xlabel('Median Salary')
    fig.savefig(file)

#for i in range(100):    
#pandas_test()
#polars_test()
pyarrow_test()
