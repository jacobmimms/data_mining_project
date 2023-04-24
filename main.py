import pandas as pd 

def main():
    columns = "user_id public completion_percentage gender region last_login registration AGE body I_am_working_in_field spoken_languages hobbies I_most_enjoy_good_food pets body_type my_eyesight eye_color hair_color hair_type completed_level_of_education favourite_color relation_to_smoking relation_to_alcohol sign_in_zodiac on_pokec_i_am_looking_for love_is_for_me relation_to_casual_sex my_partner_should_be marital_status children relation_to_children I_like_movies I_like_watching_movie I_like_music I_mostly_like_listening_to_music the_idea_of_good_evening I_like_specialties_from_kitchen fun I_am_going_to_concerts my_active_sports my_passive_sports profession I_like_books life_style music cars politics relationships art_culture hobbies_interests science_technologies computers_internet education sport movies travelling health companies_brands more"
    columns = columns.split(" ")
    ### read soc_pokec_profiles.txt into a dataframe
    df = pd.read_csv("soc-pokec-profiles.txt", sep="\t", names=columns, index_col=False)
    ## shuffle the rows of the dataframe
    df = df.sample(frac=1).reset_index(drop=True)
    ### print the first 5 rows of the dataframe
    print(df.head())
    ### save the dataframe as a 10 csv files with 1/5 of the rows in each file
    df1 = df.iloc[:100000]
    df2 = df.iloc[100000:200000]
    df3 = df.iloc[200000:300000]
    df4 = df.iloc[300000:400000]
    df5 = df.iloc[400000:500000]
    df6 = df.iloc[500000:600000]
    df7 = df.iloc[600000:700000]
    df8 = df.iloc[700000:800000]
    df9 = df.iloc[800000:900000]
    df10 = df.iloc[900000:]
    df1.to_csv("pokec1.csv", index=False)   
    df2.to_csv("pokec2.csv", index=False)
    df3.to_csv("pokec3.csv", index=False)
    df4.to_csv("pokec4.csv", index=False)
    df5.to_csv("pokec5.csv", index=False)
    df6.to_csv("pokec6.csv", index=False)
    df7.to_csv("pokec7.csv", index=False)
    df8.to_csv("pokec8.csv", index=False)
    df9.to_csv("pokec9.csv", index=False)
    df10.to_csv("pokec10.csv", index=False)

    # filter the  dataframe so only people who have completed more than 50% of their profile are included
    df = df[df["completion_percentage"] > 50]
    # save to 10 equally sized csv files
    length = len(df)
    df1 = df.iloc[:int(length/10)]
    df2 = df.iloc[int(length/10):int(length/5)]
    df3 = df.iloc[int(length/5):int(3*length/10)]
    df4 = df.iloc[int(3*length/10):int(2*length/5)]
    df5 = df.iloc[int(2*length/5):int(5*length/10)]
    df6 = df.iloc[int(5*length/10):int(3*length/5)]
    df7 = df.iloc[int(3*length/5):int(7*length/10)]
    df8 = df.iloc[int(7*length/10):int(4*length/5)]
    df9 = df.iloc[int(4*length/5):int(9*length/10)]
    df10 = df.iloc[int(9*length/10):]
    df1.to_csv("pokec1_filtered.csv", index=False)
    df2.to_csv("pokec2_filtered.csv", index=False)
    df3.to_csv("pokec3_filtered.csv", index=False)
    df4.to_csv("pokec4_filtered.csv", index=False)
    df5.to_csv("pokec5_filtered.csv", index=False)
    df6.to_csv("pokec6_filtered.csv", index=False)
    df7.to_csv("pokec7_filtered.csv", index=False)
    df8.to_csv("pokec8_filtered.csv", index=False)
    df9.to_csv("pokec9_filtered.csv", index=False)
    df10.to_csv("pokec10_filtered.csv", index=False)







if __name__ == "__main__":
    main()