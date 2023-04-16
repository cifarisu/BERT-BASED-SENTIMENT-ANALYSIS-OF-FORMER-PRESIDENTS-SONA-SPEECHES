import streamlit as st

def show_explore_page():
    st.title("Presidents")
    
    # Create a dictionary of presidents with their name, image filename, and biography text
    presidents = {
        'Joseph Ejercito Estrada': {
            'image': 'estrada.png',
            'biography': 'Joseph Ejercito Estrada is a Filipino politician and former actor who served as the 13th President of the Philippines from 1998 to 2001. He is the first person in Philippine history to be elected both President and Vice President.'
        },
        'Gloria Macapagal-Arroyo': {
            'image': 'arroyo.png',
            'biography': 'Gloria Macapagal-Arroyo is a Filipina economist and politician who served as the 14th President of the Philippines from 2001 to 2010. She is the first woman to hold the office of the President in the Philippines.'
        },
        'Benigno Aquino III': {
            'image': 'aquino.png',
            'biography': 'Benigno Simeon Cojuangco Aquino III, also known as Noynoy Aquino or PNoy, was a Filipino politician who served as the 15th President of the Philippines from 2010 until 2016. He was the third-youngest person to be elected President of the Philippines.'
        },
        'Rodrigo Duterte': {
            'image': 'duterte.png',
            'biography': 'Rodrigo Roa Duterte is a Filipino politician and lawyer who is the 16th President of the Philippines. He is known for his tough stance on crime and his controversial statements.'
        }
    }
    
    # Create two columns for each president: one for the picture and one for the biography text
    for president_name, president_info in presidents.items():
        col1, col2 = st.columns([1, 3])
        with col1:
            # Display the president's picture
            image = president_info['image']
            st.image(image, use_column_width=True)
        with col2:
            # Display the president's biography text
            biography = president_info['biography']
            st.write(president_name)
            st.write(biography, style={"text-align": "justify"})

if __name__ == '__main__':
    show_explore_page()
