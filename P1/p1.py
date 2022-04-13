import sys
from pyspark.sql import SparkSession
from pyspark.sql import functions as f
from pyspark.sql.functions import col,when,count
#feel free to def new functions if you need

def create_dataframe(filepath, format, spark):
    """
    Create a spark df given a filepath and format.

    :param filepath: <str>, the filepath
    :param format: <str>, the file format (e.g. "csv" or "json")
    :param spark: <str> the spark session

    :return: the spark df uploaded
    """

    #add your code here

    # spark_df = spark.read.format(format).load(filepath) 

    if (format == "csv") == True:
            spark_df = spark.read.csv(filepath, header=True)
    else:
        spark_df = spark.read.json(filepath)
    return spark_df


def transform_nhis_data(nhis_df):
    """
    Transform df elements

    :param nhis_df: spark df
    :return: spark df, transformed df
    """

    #add your code here
    nhis_df.dropna(how = 'any')

    transformed_df = nhis_df.withColumn('_AGEG5YR',
        f.when(f.col('AGE_P') ==  0, 0)
         .when(f.col('AGE_P') <= 24, 1)
         .when(f.col('AGE_P') <= 29, 2)
         .when(f.col('AGE_P') <= 34, 3)
         .when(f.col('AGE_P') <= 39, 4)
         .when(f.col('AGE_P') <= 44, 5)
         .when(f.col('AGE_P') <= 49, 6)
         .when(f.col('AGE_P') <= 54, 7)
         .when(f.col('AGE_P') <= 59, 8)
         .when(f.col('AGE_P') <= 64, 9)
         .when(f.col('AGE_P') <= 69, 10)
         .when(f.col('AGE_P') <= 74, 11)
         .when(f.col('AGE_P') <= 79, 12)
         .when(f.col('AGE_P') <= 84, 13)
         .otherwise(14))

    transformed_df = transformed_df.withColumn('_IMPRACE',
        f.when(f.col("HISPAN_I") <  9, 5)
         .when(f.col("MRACBPI2") == 1, 1)
         .when(f.col("MRACBPI2") == 2, 2)
         .when(f.col("MRACBPI2") == 3, 4)
         .when((f.col("MRACBPI2") == 6) | (f.col("MRACBPI2") == 7) | (f.col("MRACBPI2") == 12), 3)
         .when(f.col("MRACBPI2") == 16,6)
         .otherwise(6))
    transformed_df = transformed_df.drop("HISPAN_I", "MRACBPI2",'AGE_P')
    
    return transformed_df


def calculate_statistics(joined_df):
    """
    Calculate prevalence statistics

    :param joined_df: the joined df

    :return: None
    """
    #add your code here
   
    sex_df = joined_df.groupBy("SEX").agg(
    count(when(f.col("DIBEV1") == '1', True)).alias('Count_Yes'),
    count(when(f.col("DIBEV1") != '1', True)).alias('Count_No')
    )

    age_df = joined_df.groupBy("_AGEG5YR").agg(
    count(when(f.col("DIBEV1") == '1', True)).alias('Count_Yes'),
    count(when(f.col("DIBEV1") != '1', True)).alias('Count_No')
    )

    race_df = joined_df.groupBy("_IMPRACE").agg(
    count(when(f.col("DIBEV1") == '1', True)).alias('Count_Yes'),
    count(when(f.col("DIBEV1") != '1', True)).alias('Count_No')
    )

    sex_df = sex_df.withColumn('Sex_Prevalence',f.col('Count_Yes')/(f.col('Count_Yes')+f.col('Count_No'))).show()
    age_df = age_df.withColumn('Age_Prevalence',f.col('Count_Yes')/(f.col('Count_Yes')+f.col('Count_No'))).show()
    race_df = race_df.withColumn('Race_Prevalence',f.col('Count_Yes')/(f.col('Count_Yes')+f.col('Count_No'))).show()



    # raceBook = {"White, Non-Hispanic":0,
    #             "Black, Non-Hispanic":0,
    #             "Asian, Non-Hispanic":0,
    #             "American Indian/Alaskan Native, Non-Hispanic":0,
    #             "Hispanic":0,
    #             "Other race, Non-Hispanic":0}
    # genderBook = {"Male":0,
    #               "Female":0}
    # ageBook = {"18-24":0,
    #            "25-29":0,
    #            "30-34":0,
    #            "35-39":0,
    #            "40-44":0,
    #            "45-49":0,
    #            "50-54":0,
    #            "55-59":0,
    #            "60-64":0,
    #            "65-69":0,
    #            "70-74":0,
    #            "75-79":0,
    #            ">80":0,
    #            "Missing":0}

    # for entry in joined_df['SEX']:
    #     if entry == 1:
    #         genderBook['Male']+=1
    #     else:
    #         genderBook['Female']+=1

    # for entry in joined_df['_IMPRACE']:
    #     if entry == 1:
    #         raceBook["White, Non-Hispanic"]+=1
    #     elif entry == 2:
    #         raceBook["Black, Non-Hispanic"]+=1
    #     elif entry == 3:
    #         raceBook["Asian, Non-Hispanic"]+=1
    #     elif entry == 4:
    #         raceBook["American Indian/Alaskan Native, Non-Hispanic"]+=1
    #     elif entry == 5:
    #         raceBook["Hispanic"]+=1
    #     elif entry == 6:
    #         raceBook["Other race, Non-Hispanic"]+=1

    # for entry in joined_df['_AGEG5YR']:
    #     if entry == 1:
    #         ageBook["18-24"]+=1
    #     elif entry == 2:
    #         ageBook["25-29"]+=1
    #     elif entry == 3:
    #         ageBook["30-34"]+=1
    #     elif entry == 4:
    #         ageBook["35-39"]+=1
    #     elif entry == 5:
    #         ageBook["40-44"]+=1
    #     elif entry == 6:
    #         ageBook["45-49"]+=1
    #     elif entry == 7:
    #         ageBook["50-54"]+=1
    #     elif entry == 8:
    #         ageBook["55-59"]+=1
    #     elif entry == 9:
    #         ageBook["60-64"]+=1
    #     elif entry == 10:
    #         ageBook["65-69"]+=1
    #     elif entry == 11:
    #         ageBook["70-74"]+=1
    #     elif entry == 12:
    #         ageBook["75-79"]+=1
    #     elif entry == 13:
    #         ageBook[">80"]+=1
    #     elif entry == 14:
    #         ageBook["Missing"]+=1

    #     total = len(joined_df)
        
    #     for value in genderBook.values():
    #         value = value/total
        
    #     for value in raceBook.values():
    #         value = value/total
    #     for value in ageBook.values():
    #         value = value/total

    #     return genderBook, raceBook, ageBook
        

      
    

def join_data(brfss_df, nhis_df):
    """
    Join dataframes

    :param brfss_df: spark df
    :param nhis_df: spark df after transformation
    :return: the joined df

    """
    #add your code here
    
    joined_df = nhis_df.join(brfss_df, ['SEX', '_IMPRACE', '_AGEG5YR'])

    return joined_df

if __name__ == '__main__':

    brfss_filename = sys.argv[1]
    nhis_filename = sys.argv[2]
    save_output = sys.argv[3]
    if save_output == "True":
        output_filename = sys.argv[4]
    else:
        output_filename = None
    

    # Start spark session
    spark = SparkSession.builder.getOrCreate()


    

    # Load dataframes
    brfss_df = create_dataframe(brfss_filename, 'json', spark)
    nhis_df = create_dataframe(nhis_filename, 'csv', spark)
    # Perform mapping on nhis dataframe
    nhis_df = transform_nhis_data(nhis_df)
    # Join brfss and nhis df
    joined_df = join_data(brfss_df, nhis_df)
    # Save
    if save_output == "True":
        joined_df.write.csv(output_filename, mode='overwrite', header=True)
    # Calculate and print statistics
    calculate_statistics(joined_df)

    
    
    # Stop spark session
    spark.stop()

