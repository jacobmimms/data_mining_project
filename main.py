import pandas as pd 

def main():
    columns = "user_id public completion_percentage gender region last_login registration AGE body I_am_working_in_field spoken_languages hobbies I_most_enjoy_good_food pets body_type my_eyesight eye_color hair_color hair_type completed_level_of_education favourite_color relation_to_smoking relation_to_alcohol sign_in_zodiac on_pokec_i_am_looking_for love_is_for_me relation_to_casual_sex my_partner_should_be marital_status children relation_to_children I_like_movies I_like_watching_movie I_like_music I_mostly_like_listening_to_music the_idea_of_good_evening I_like_specialties_from_kitchen fun I_am_going_to_concerts my_active_sports my_passive_sports profession I_like_books life_style music cars politics relationships art_culture hobbies_interests science_technologies computers_internet education sport movies travelling health companies_brands more"
    columns = columns.split(" ")
    ### read soc_pokec_profiles.txt into a dataframe
    df = pd.read_csv("soc-pokec-profiles.txt", sep="\t", names=columns, index_col=False)
    ## shuffle the rows of the dataframe
    df = df.sample(frac=1).reset_index(drop=True)
    length = len(df)
    num_partitions = 10
    partition_len = length / num_partitions 
    for i in range(num_partitions):
        df = df.iloc[int(i*partition_len):int(i+1*partition_len)]
        df.to_csv(f"data/pokec{i+1}.csv", index=False)

    #filter the  dataframe so only people who have completed more than 50% of their profile are included
    df = df[df["completion_percentage"] > 50]
    length = len(df)
    num_partitions = 10
    partition_len = length / num_partitions 
    for i in range(num_partitions):
        df = df.iloc[int(i*partition_len):int(i+1*partition_len)]
        df.to_csv(f"filtered_data/pokec{i+1}.csv", index=False)



if __name__ == "__main__":
    main()