import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import requests
from io import BytesIO
import random
import re
from datetime import datetime
# Add new imports for machine learning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

# Function to get high resolution poster
def get_high_res_poster(url):
    if pd.isna(url):
        return url
    # Remove size constraints from URL
    high_res_url = re.sub(r'UX\d+_CR\d+,\d+,\d+,\d+_', '', url)
    high_res_url = re.sub(r'UY\d+_CR\d+,\d+,\d+,\d+_', '', high_res_url)
    return high_res_url

# Function to classify user intent using machine learning
def classify_user_intent(user_input):
    """
    Classifies the user's input into different intents using a simple Naive Bayes classifier.
    
    Args:
        user_input (str): The user's message
        
    Returns:
        str: The classified intent (genre_recommendation, mood_recommendation, year_recommendation, or rating_recommendation)
    """
    # Define common intents and their keywords
    intents = {
        'genre_recommendation': ['genre', 'type', 'kind', 'category', 'action', 'comedy', 'drama', 'horror'],
        'mood_recommendation': ['mood', 'feel', 'feeling', 'happy', 'sad', 'excited', 'relaxed'],
        'year_recommendation': ['year', 'recent', 'old', 'new', 'classic', 'modern', 'from', 'in', 'during'],
        'rating_recommendation': ['rating', 'score', 'best', 'top', 'highest', 'rated']
    }
    
    # Create training data
    X_train = []
    y_train = []
    
    for intent, keywords in intents.items():
        for keyword in keywords:
            X_train.append(keyword)
            y_train.append(intent)
    
    # Create and train the classifier
    classifier = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('clf', MultinomialNB())
    ])
    
    classifier.fit(X_train, y_train)
    
    # Predict intent
    intent = classifier.predict([user_input.lower()])[0]
    return intent

# Dictionary of genres and their related keywords
genre_keywords = {
    'Action': ["action", "adventure", "fight", "battle", "war", "superhero", "explosion", "thriller", "exciting"],
    'Comedy': ["comedy", "funny", "laugh", "humor", "joke", "hilarious", "sitcom", "parody", "romantic comedy"],
    'Drama': ["drama", "emotional", "serious", "deep", "story", "powerful", "intense", "relationship", "life"],
    'Horror': ["horror", "scary", "fear", "thriller", "supernatural", "ghost", "monster", "suspense", "terror"],
    'Romance': ["romance", "love", "romantic", "relationship", "dating", "couple", "wedding", "passion", "heartfelt"],
    'Sci-Fi': ["sci-fi", "science fiction", "space", "future", "alien", "robot", "technology", "dystopian", "cyberpunk"],
    'Fantasy': ["fantasy", "magic", "mythical", "dragon", "wizard", "supernatural", "fairy tale", "enchanted"],
    'Crime': ["crime", "detective", "mystery", "thriller", "police", "investigation", "murder", "heist", "gangster"],
    'Animation': ["animation", "animated", "cartoon", "pixar", "disney", "family", "kids", "children"],
    'Documentary': ["documentary", "real", "true story", "historical", "educational", "biography", "nature"],
    'Musical': ["musical", "music", "singing", "dance", "song", "broadway", "performance"],
    'Western': ["western", "cowboy", "wild west", "frontier", "desert", "gunslinger"],
    'Sports': ["sports", "athlete", "game", "competition", "football", "baseball", "basketball", "boxing"],
    'War': ["war", "military", "soldier", "battle", "army", "combat", "historical war", "world war"],
    'Thriller': ["thriller", "suspense", "mystery", "tension", "psychological", "crime thriller", "suspenseful"],
    'Family': ["family", "kids", "children", "parents", "heartwarming", "wholesome", "feel-good"]
}

# Mood-based recommendations
mood_keywords = {
    'Feel-good': ["happy", "uplifting", "positive", "cheerful", "feel good", "heartwarming", "inspiring", "joyful", "lighthearted"],
    'Dark': ["dark", "gritty", "intense", "disturbing", "psychological", "twisted", "noir", "sad", "depressing", "melancholic", "emotional", "tearjerker", "heartbreaking", "tragic", "sorrowful", "gloomy", "miserable", "unhappy", "blue", "down", "low"],
    'Inspirational': ["inspiring", "motivation", "uplifting", "true story", "achievement", "success", "empowering", "encouraging"],
    'Thought-provoking': ["thought provoking", "philosophical", "deep", "meaningful", "intellectual", "complex", "mind-bending", "challenging"],
    'Relaxing': ["relaxing", "calm", "peaceful", "easy watch", "light hearted", "gentle", "soothing", "tranquil"],
    'Exciting': ["exciting", "thrilling", "action packed", "adventure", "adrenaline", "fast paced", "intense", "gripping"]
}

# Function to generate chat response with enhanced classification
def generate_chat_response(user_input, df):
    """
    Generates a movie recommendation based on user input using intent classification.
    
    Args:
        user_input (str): The user's message
        df (DataFrame): The movie dataset
        
    Returns:
        str: A personalized movie recommendation
    """
    try:
        # Check for greetings first
        greetings = ['hello', 'hi', 'hey', 'greetings', 'good morning', 'good afternoon', 'good evening']
        if user_input.lower() in greetings:
            return "Hello! I would love to recommend you a movie to help you have a great day!!! 🎬\n\nJust let me know what kind of movie you're looking for - I can help with genres, moods, years, or ratings! 😊"
        
        # Get ML classification
        intent = classify_user_intent(user_input)
        
        # Initialize flags for matching
        has_matched_genre = False
        has_matched_mood = False
        has_matched_year = False
        has_matched_rating = False
        
        # Get matched genres
        matched_genres = []
        for genre, keywords in genre_keywords.items():
            if any(word in user_input.lower() for word in keywords):
                has_matched_genre = True
                matched_genres.append(genre)
        
        # Get matched moods
        matched_moods = []
        for mood, keywords in mood_keywords.items():
            if any(word in user_input.lower() for word in keywords):
                has_matched_mood = True
                matched_moods.append(mood)
        
        # Check for year matches
        year_match = re.search(r'\b(19|20)\d{2}\b', user_input)
        if year_match or any(word in user_input.lower() for word in ["recent", "new", "latest", "modern", "current", "classic", "old", "vintage", "retro", "timeless"]):
            has_matched_year = True
        
        # Check for rating matches
        if any(word in user_input.lower() for word in ["best", "top", "highest", "greatest", "amazing", "rating", "score"]):
            has_matched_rating = True
        
        # If no matches found, return apology message
        if not (has_matched_genre or has_matched_mood or has_matched_year or has_matched_rating):
            return "I am sorry I couldn't help you with that, should I recommend a movie? 😊"
        
        # Start with the full dataset
        filtered_df = df.copy()
        
        # Apply year filter if present
        if year_match:
            requested_year = int(year_match.group())
            filtered_df = filtered_df[filtered_df['Released_Year'] == requested_year]
            if filtered_df.empty:
                return f"I couldn't find any movies from {requested_year} in our database. Would you like a recommendation from a different year?"
        elif any(word in user_input.lower() for word in ["recent", "new", "latest", "modern", "current"]):
            filtered_df = filtered_df[filtered_df['Released_Year'] >= 2010]
        elif any(word in user_input.lower() for word in ["classic", "old", "vintage", "retro", "timeless"]):
            filtered_df = filtered_df[filtered_df['Released_Year'] < 1990]
        
        # Apply rating filter if present
        if has_matched_rating and any(word in user_input.lower() for word in ["best", "top", "highest", "greatest", "amazing"]):
            filtered_df = filtered_df[filtered_df['IMDB_Rating'] >= 8.0]
        
        # Apply genre filter if present
        if has_matched_genre and not filtered_df.empty:
            genre_filter = filtered_df['Genre'].str.contains('|'.join(matched_genres), case=False, na=False)
            filtered_df = filtered_df[genre_filter]
            if filtered_df.empty:
                return f"I couldn't find any {', '.join(matched_genres)} movies matching your criteria. Would you like to try a different genre or year?"
        
        # Apply mood-based filtering
        if has_matched_mood and not filtered_df.empty:
            # For sad/dark moods, prefer dramas and thrillers with lower ratings
            if matched_moods[0] in ['Dark']:
                filtered_df = filtered_df[filtered_df['Genre'].str.contains('Drama|Thriller|Crime|Horror', case=False, na=False)]
                # Sort by rating in ascending order for sad movies
                filtered_df = filtered_df.sort_values('IMDB_Rating', ascending=True)
            # For feel-good moods, prefer comedies and family movies with higher ratings
            elif matched_moods[0] in ['Feel-good']:
                filtered_df = filtered_df[filtered_df['Genre'].str.contains('Comedy|Family|Romance|Musical', case=False, na=False)]
                filtered_df = filtered_df.sort_values('IMDB_Rating', ascending=False)
            # For exciting moods, prefer action and adventure movies
            elif matched_moods[0] in ['Exciting']:
                filtered_df = filtered_df[filtered_df['Genre'].str.contains('Action|Adventure|Thriller|Sci-Fi', case=False, na=False)]
                filtered_df = filtered_df.sort_values('IMDB_Rating', ascending=False)
            # For relaxing moods, prefer comedies and family movies
            elif matched_moods[0] in ['Relaxing']:
                filtered_df = filtered_df[filtered_df['Genre'].str.contains('Comedy|Family|Romance|Animation', case=False, na=False)]
                filtered_df = filtered_df.sort_values('IMDB_Rating', ascending=False)
            # For inspirational moods, prefer dramas and biographies
            elif matched_moods[0] in ['Inspirational']:
                filtered_df = filtered_df[filtered_df['Genre'].str.contains('Drama|Biography|Sport|History', case=False, na=False)]
                filtered_df = filtered_df.sort_values('IMDB_Rating', ascending=False)
            # For thought-provoking moods, prefer dramas and mysteries
            elif matched_moods[0] in ['Thought-provoking']:
                filtered_df = filtered_df[filtered_df['Genre'].str.contains('Drama|Sci-Fi|Mystery|Thriller', case=False, na=False)]
                filtered_df = filtered_df.sort_values('IMDB_Rating', ascending=False)
        
        # If we have no movies after filtering, return a helpful message
        if filtered_df.empty:
            return "I couldn't find any movies matching all your criteria. Would you like to try different filters?"
        
        # Get a random movie from the top 5 matches
        top_movies = filtered_df.head(5)
        movie = top_movies.sample(1).iloc[0]
        
        # Generate response based on the filters used and ML classification
        response_parts = []
        
        # Add genre/mood information based on ML classification
        if intent == 'genre_recommendation' and has_matched_genre:
            genre_text = ' and '.join(matched_genres)
            response_parts.append(f"a great {genre_text} movie")
        elif intent == 'mood_recommendation' and has_matched_mood:
            response_parts.append(f"a {matched_moods[0].lower()} experience")
        elif has_matched_genre:
            genre_text = ' and '.join(matched_genres)
            response_parts.append(f"a great {genre_text} movie")
        elif has_matched_mood:
            response_parts.append(f"a {matched_moods[0].lower()} experience")
        
        # Add year information
        if intent == 'year_recommendation' or year_match:
            response_parts.append(f"from {int(movie['Released_Year'])}")
        
        # Add rating information
        if intent == 'rating_recommendation' or has_matched_rating:
            response_parts.append("highly rated")
        
        # Construct the final response
        response = f"I recommend '{movie['Series_Title']}' ({int(movie['Released_Year'])}). It's {' '.join(response_parts)} with a rating of {movie['IMDB_Rating']}. "
        
        if pd.notna(movie['Overview']):
            response += f"\nHere's what it's about: {movie['Overview']}"
        
        return response
    
    except Exception as e:
        return "I apologize, but I'm having trouble generating a recommendation right now. Please try again."

# Set page config
st.set_page_config(
    page_title="🎬 Movie Magic Dashboard",
    page_icon="🎬",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .movie-card {
        background-color: #1a1a1a;
        border-radius: 10px;
        padding: 15px;
        margin: 10px;
        transition: transform 0.3s;
    }
    .movie-card:hover {
        transform: scale(1.02);
    }
    .movie-title {
        color: #ffffff;
        font-size: 1.2em;
        font-weight: bold;
    }
    .movie-info {
        color: #b3b3b3;
        font-size: 0.9em;
    }
    .stButton>button {
        background-color: #ff4b4b;
        color: white;
        border-radius: 5px;
        padding: 10px 20px;
        border: none;
    }
    .stButton>button:hover {
        background-color: #ff3333;
    }
    .filter-section {
        background-color: #1a1a1a;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv('imdb_top_1000.csv')
    # Clean data
    df['Released_Year'] = pd.to_numeric(df['Released_Year'], errors='coerce')
    df['IMDB_Rating'] = pd.to_numeric(df['IMDB_Rating'], errors='coerce')
    df['Meta_score'] = pd.to_numeric(df['Meta_score'], errors='coerce')
    # Convert poster URLs to high resolution
    df['Poster_Link'] = df['Poster_Link'].apply(get_high_res_poster)
    # Remove any rows with missing values in key columns
    df = df.dropna(subset=['Released_Year', 'IMDB_Rating'])
    return df

df = load_data()

# Main content
st.markdown("<h1 style='text-align: center;'>🎬 FilmFusion</h1>", unsafe_allow_html=True)
st.markdown("---")

# Filter section
st.markdown('<div class="filter-section">', unsafe_allow_html=True)
st.write("### 🎯 Filters")

# Create columns for filters
col1, col2, col3 = st.columns(3)

with col1:
    # Genre filter
    all_genres = sorted(set(genre for genres in df['Genre'].str.split(', ') for genre in genres))
    selected_genres = st.multiselect("Select Genres", all_genres)

with col2:
    # Year range filter
    min_year, max_year = int(df['Released_Year'].min()), int(df['Released_Year'].max())
    year_range = st.slider("Select Year Range", min_year, max_year, (min_year, max_year))

with col3:
    # Rating filter with proper handling of NaN values
    valid_ratings = df['IMDB_Rating'].dropna()
    if not valid_ratings.empty:
        min_rating, max_rating = float(valid_ratings.min()), float(valid_ratings.max())
        rating_range = st.slider("Select Rating Range", min_rating, max_rating, (min_rating, max_rating))
    else:
        st.warning("No valid ratings found in the dataset")
        rating_range = (0.0, 10.0)

st.markdown('</div>', unsafe_allow_html=True)

# Filter data with proper handling of NaN values and closest matches
filtered_df = df[
    (df['Released_Year'].between(year_range[0], year_range[1]))
]

# Handle rating filter with closest matches
if not filtered_df.empty:
    valid_ratings = filtered_df['IMDB_Rating'].dropna()
    if not valid_ratings.empty:
        # If no movies in the exact range, find the closest matches
        if not filtered_df['IMDB_Rating'].between(rating_range[0], rating_range[1]).any():
            # Find the closest rating above the selected range
            closest_above = valid_ratings[valid_ratings >= rating_range[0]].min()
            if pd.notna(closest_above):
                st.info(f"No movies found in the selected rating range. Showing movies with rating closest to your selection.")
                filtered_df = filtered_df[filtered_df['IMDB_Rating'] >= rating_range[0]]
            else:
                # If no movies above, find the closest below
                closest_below = valid_ratings[valid_ratings <= rating_range[1]].max()
                if pd.notna(closest_below):
                    st.info(f"No movies found in the selected rating range. Showing movies with rating closest to your selection.")
                    filtered_df = filtered_df[filtered_df['IMDB_Rating'] <= rating_range[1]]
        else:
            # If there are movies in the exact range, use that
            filtered_df = filtered_df[filtered_df['IMDB_Rating'].between(rating_range[0], rating_range[1])]

# Apply genre filter if genres are selected
if selected_genres:
    filtered_df = filtered_df[filtered_df['Genre'].apply(lambda x: any(genre in x for genre in selected_genres))]

# Top section with stats
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Movies", len(filtered_df))
with col2:
    st.metric("Average Rating", f"{filtered_df['IMDB_Rating'].mean():.1f}")
with col3:
    st.metric("Oldest Movie", int(filtered_df['Released_Year'].min()))
with col4:
    st.metric("Newest Movie", int(filtered_df['Released_Year'].max()))

# Tabs
tab1, tab2, tab3 = st.tabs(["🎥 Movie Gallery", "📊 Statistics", "🎯 Recommendations"])

with tab1:
    st.subheader("Movie Gallery")
    search_query = st.text_input("Search Movies", "")
    
    if search_query:
        search_df = filtered_df[filtered_df['Series_Title'].str.contains(search_query, case=False)]
    else:
        search_df = filtered_df
    
    # Display movies in a grid
    cols = st.columns(4)
    for idx, movie in search_df.iterrows():
        with cols[idx % 4]:
            st.markdown(f"""
                <div class="movie-card">
                    <img src="{movie['Poster_Link']}" style="width:100%; border-radius:5px; margin-bottom:10px; object-fit: cover; height: 400px;">
                    <div class="movie-title">{movie['Series_Title']} ({int(movie['Released_Year'])})</div>
                    <div class="movie-info">⭐ {movie['IMDB_Rating']} | {movie['Runtime']}</div>
                    <div class="movie-info">{movie['Genre']}</div>
                </div>
            """, unsafe_allow_html=True)

with tab2:
    st.subheader("Movie Statistics")
    
    # Check if we have any data to display
    if filtered_df.empty:
        st.warning("No movies match the selected filters. Please adjust your filters to see statistics.")
    else:
        try:
            # Add ML Classification Visualization
            st.write("### 🤖 Machine Learning Classification Analysis")
            
            # Create sample queries for each intent
            sample_queries = {
                'Genre Recommendations': [
                    "I want to watch a comedy",
                    "Show me some action movies",
                    "Any drama recommendations?",
                    "Looking for horror films"
                ],
                'Mood Recommendations': [
                    "I'm feeling sad",
                    "Want something exciting",
                    "Need a feel-good movie",
                    "Looking for something relaxing"
                ],
                'Year Recommendations': [
                    "Movies from 2020",
                    "Classic films from 1990",
                    "Recent releases",
                    "Old movies from 1980"
                ],
                'Rating Recommendations': [
                    "Best rated movies",
                    "Top rated films",
                    "Highest rated action movies",
                    "Show me amazing movies"
                ]
            }
            
            # Create a DataFrame for visualization
            intent_data = []
            for category, queries in sample_queries.items():
                for query in queries:
                    intent = classify_user_intent(query)
                    intent_data.append({
                        'Category': category,
                        'Intent': intent,
                        'Query': query
                    })
            
            intent_df = pd.DataFrame(intent_data)
            
            # Create a beautiful visualization using plotly
            import plotly.express as px
            import plotly.graph_objects as go
            
            # Create a sunburst chart
            fig = px.sunburst(
                intent_df,
                path=['Category', 'Intent'],
                title='Movie Recommendation Classification Analysis',
                color='Category',
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            
            # Update layout for better appearance
            fig.update_layout(
                title_x=0.5,
                title_font_size=20,
                height=600,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white')
            )
            
            # Display the plot
            st.plotly_chart(fig, use_container_width=True)
            
            # Add explanation
            st.markdown("""
            <div style='background-color: #1a1a1a; padding: 20px; border-radius: 10px; margin-top: 20px;'>
                <h4 style='color: #ffffff;'>Understanding the Classification</h4>
                <p style='color: #b3b3b3;'>
                This visualization shows how our machine learning model classifies different types of movie requests:
                </p>
                <ul style='color: #b3b3b3;'>
                    <li><strong>Genre Recommendations:</strong> When users ask for specific types of movies</li>
                    <li><strong>Mood Recommendations:</strong> When users want movies based on their current mood</li>
                    <li><strong>Year Recommendations:</strong> When users ask for movies from specific time periods</li>
                    <li><strong>Rating Recommendations:</strong> When users want to see highly-rated movies</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
            # Original statistics code continues here...
            # Rating distribution
            st.write("### Rating Distribution")
            rating_data = filtered_df['IMDB_Rating'].dropna()
            if not rating_data.empty:
                rating_counts = rating_data.value_counts().sort_index()
                st.bar_chart(
                    rating_counts,
                    use_container_width=True,
                    color="#1E90FF"  # Dodger Blue color for rating distribution
                )
            else:
                st.info("No rating data available for the selected filters.")
            
            # Year vs Rating
            st.write("### Year vs Rating")
            year_rating_data = filtered_df[['Released_Year', 'IMDB_Rating']].dropna()
            if not year_rating_data.empty:
                year_rating_chart = pd.DataFrame(year_rating_data.groupby('Released_Year')['IMDB_Rating'].mean())
                st.line_chart(
                    year_rating_chart,
                    use_container_width=True,
                    color="#FFA500"  # Orange color for year vs rating
                )
            else:
                st.info("No year-rating data available for the selected filters.")
            
            # Genre distribution
            st.write("### Genre Distribution")
            if not filtered_df.empty and 'Genre' in filtered_df.columns:
                genre_list = []
                for genres in filtered_df['Genre'].dropna():
                    genre_list.extend([g.strip() for g in genres.split(',')])
                genre_counts = pd.Series(genre_list).value_counts()
                if not genre_counts.empty:
                    st.bar_chart(
                        genre_counts,
                        use_container_width=True,
                        color="#4CAF50"  # Green color for genre distribution
                    )
                else:
                    st.info("No genre data available for the selected filters.")
            
            # Additional statistics
            st.write("### Additional Statistics")
            col1, col2 = st.columns(2)
            
            with col1:
                # Top rated movies
                st.write("#### Top Rated Movies")
                if not filtered_df.empty:
                    top_movies = filtered_df.nlargest(5, 'IMDB_Rating')[['Series_Title', 'IMDB_Rating', 'Released_Year']]
                    if not top_movies.empty:
                        # Format the data before displaying
                        formatted_top_movies = top_movies.copy()
                        formatted_top_movies['IMDB_Rating'] = formatted_top_movies['IMDB_Rating'].round(1)
                        formatted_top_movies['Released_Year'] = formatted_top_movies['Released_Year'].astype(int)
                        
                        # Rename columns for better display
                        formatted_top_movies.columns = ['Movie Title', 'Rating', 'Year']
                        
                        # Display the table with basic formatting
                        st.dataframe(
                            formatted_top_movies,
                            hide_index=True,
                            column_config={
                                "Movie Title": st.column_config.TextColumn(
                                    "Movie Title",
                                    width="large"
                                ),
                                "Rating": st.column_config.NumberColumn(
                                    "Rating",
                                    format="%.1f ⭐"
                                ),
                                "Year": st.column_config.NumberColumn(
                                    "Year",
                                    format="%d"
                                )
                            }
                        )
                    else:
                        st.info("No movie ratings available.")
            
            with col2:
                # Movies by Year
                st.write("#### Movies by Year")
                if not filtered_df.empty:
                    movies_by_year = filtered_df['Released_Year'].value_counts().sort_index()
                    if not movies_by_year.empty:
                        st.line_chart(
                            movies_by_year,
                            use_container_width=True,
                            color="#9C27B0"  # Purple color for movies by year
                        )
                    else:
                        st.info("No year data available.")

        except Exception as e:
            st.error("An error occurred while generating statistics. Please try adjusting your filters.")

with tab3:
    st.subheader("Movie Recommendations")
    
    # Create two columns for the recommender section with 60-40 split
    col1, col2 = st.columns([0.6, 0.4])
    
    with col1:
        # Chat interface with centered heading
        st.markdown("<h3 style='text-align: center;'>🤖 Movie Bot</h3>", unsafe_allow_html=True)
        st.write("Ask me for movie recommendations or ask questions about movies!")
        
        # Create a container for messages
        chat_container = st.container()
        
        # Create a container for input at the bottom
        input_container = st.container()
        
        # Handle input first (but it will appear at the bottom)
        with input_container:
            user_input = st.chat_input("Ask for movie recommendations...")
        
        # Initialize chat history if it doesn't exist
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        
        # Display chat history in the messages container
        with chat_container:
            for message in st.session_state.chat_history:
                with st.chat_message(message["role"]):
                    st.write(message["content"])
            
            # Display new messages
            if user_input:
                # Display user message
                with st.chat_message("user"):
                    st.write(user_input)
                
                # Generate and display assistant response
                with st.chat_message("assistant"):
                    response = generate_chat_response(user_input, filtered_df)
                    st.write(response)
                
                # Update chat history
                st.session_state.chat_history.extend([
                    {"role": "user", "content": user_input},
                    {"role": "assistant", "content": response}
                ])
    
    with col2:
        # Random movie section with centered heading
        st.markdown("<h3 style='text-align: center;'>🎲 Random Movie</h3>", unsafe_allow_html=True)
        
        # Center the button
        col1, col2, col3 = st.columns([1,2,1])
        with col2:
            if st.button("Get Random Movie"):
                random_movie = filtered_df.sample(1).iloc[0]
                st.markdown(f"""
                    <div class="movie-card" style="width: 100%; max-width: none; margin: 0 auto;">
                        <img src="{random_movie['Poster_Link']}" style="width:100%; border-radius:5px; margin-bottom:10px; object-fit: cover; height: 400px;">
                        <div class="movie-title" style="font-size: 1.4em;">{random_movie['Series_Title']} ({int(random_movie['Released_Year'])})</div>
                        <div class="movie-info" style="font-size: 1.2em;">⭐ {random_movie['IMDB_Rating']} | {random_movie['Runtime']}</div>
                        <div class="movie-info" style="font-size: 1.2em;">{random_movie['Genre']}</div>
                        <div class="movie-info" style="margin-top:10px; font-size: 1.2em;">{random_movie['Overview']}</div>
                    </div>
                """, unsafe_allow_html=True)
        
        # Similar movies based on genre
        if selected_genres:
            st.markdown("<h3 style='text-align: center;'>Similar Movies</h3>", unsafe_allow_html=True)
            similar_movies = filtered_df[filtered_df['Genre'].apply(
                lambda x: any(genre in x for genre in selected_genres)
            )].sort_values('IMDB_Rating', ascending=False).head(5)
            
            for _, movie in similar_movies.iterrows():
                st.markdown(f"""
                    <div class="movie-card" style="width: 100%; max-width: none; margin: 0 auto;">
                        <div class="movie-title" style="font-size: 1.3em;">{movie['Series_Title']} ({int(movie['Released_Year'])})</div>
                        <div class="movie-info" style="font-size: 1.2em;">⭐ {movie['IMDB_Rating']} | {movie['Runtime']}</div>
                        <div class="movie-info" style="font-size: 1.2em;">{movie['Genre']}</div>
                    </div>
                """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("Data from IMDb Top 1000 Movies") 