import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

# Spotify API credentials
sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id="your_client_id",
                                                           client_secret="your_client_secret"))

# Fetch songs based on emotion
def get_songs_by_emotion(emotion):
    if emotion == 'happy':
        playlist_id = 'spotify_playlist_id_for_happy_songs'
    elif emotion == 'sad':
        playlist_id = 'spotify_playlist_id_for_sad_songs'
    # Add more emotions as needed

    results = sp.playlist_tracks(playlist_id)
    songs = results['items']
    return [song['track']['name'] for song in songs]

# Example usage
emotion = 'happy'
recommended_songs = get_songs_by_emotion(emotion)
print(f"Recommended songs for {emotion} emotion: {recommended_songs}")
