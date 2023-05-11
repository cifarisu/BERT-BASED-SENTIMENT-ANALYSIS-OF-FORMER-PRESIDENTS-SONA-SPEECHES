import streamlit as st

def show_explore_page():
    st.title("Presidents")
    
    # Create a dictionary of presidents with their name, image filename, and biography text
    presidents = {
        'JOSEPH EJERCITO ESTRADA': {
            'image': 'estrada.png',
            'biography': 'JOSEPH EJERCITO ESTRADA IS A FILIPINO POLITICIAN AND FORMER ACTOR WHO SERVED AS THE 13TH PRESIDENT OF THE PHILIPPINES FROM 1998 TO 2001. HE IS THE FIRST PERSON IN PHILIPPINE HISTORY TO BE ELECTED BOTH PRESIDENT AND VICE PRESIDENT.'
        },
        'GLORIA MACAPAGAL-ARROYO': {
            'image': 'arroyo.png',
            'biography': 'GLORIA MACAPAGAL-ARROYO IS A FILIPINA ECONOMIST AND POLITICIAN WHO SERVED AS THE 14TH PRESIDENT OF THE PHILIPPINES FROM 2001 TO 2010. SHE IS THE FIRST WOMAN TO HOLD THE OFFICE OF THE PRESIDENT IN THE PHILIPPINES.'
        },
        'BENIGNO AQUINO III': {
            'image': 'aquino.png',
            'biography': 'BENIGNO SIMEON COJUANGCO AQUINO III, ALSO KNOWN AS NOYNOY AQUINO OR PNOY, WAS A FILIPINO POLITICIAN WHO SERVED AS THE 15TH PRESIDENT OF THE PHILIPPINES FROM 2010 UNTIL 2016. HE WAS THE THIRD-YOUNGEST PERSON TO BE ELECTED PRESIDENT OF THE PHILIPPINES.'
        },
        'RODRIGO DUTERTE': {
            'image': 'duterte.png',
            'biography': 'RODRIGO ROA DUTERTE IS A FILIPINO POLITICIAN AND LAWYER WHO IS THE 16TH PRESIDENT OF THE PHILIPPINES. HE IS KNOWN FOR HIS TOUGH STANCE ON CRIME AND HIS CONTROVERSIAL STATEMENTS.'
        }
    }
    
    # Initialize the SessionState variable for current president index
    if 'current_president_index' not in st.session_state:
        st.session_state.current_president_index = 0
    
    # Create left and right arrow buttons
    col1, col2, col3, col4, col5 = st.columns([1, 1, 2, 1, 1])
    with col1:
        st.write(' ')
    with col2:
        show_previous_button = st.button(label='&larr;', help='Show previous president', key='prev_button')
    with col3:
        st.write(' ')
    with col4:
        st.write(' ')
    with col5:
        show_next_button = st.button(label='&rarr;', help='Show next president', key='next_button')


    # Update the current president index when the "Show next president" or "Show previous president" button is clicked
    if show_previous_button:
        st.session_state.current_president_index = (st.session_state.current_president_index - 1) % len(presidents)
    elif show_next_button:
        st.session_state.current_president_index = (st.session_state.current_president_index + 1) % len(presidents)
    # Create two columns for each president: one for the picture and one for the biography text
    col1, col2, col3 = st.columns(3)

    with col1:
        st.write(' ')
    with col2:
            # Display the current president's picture
        image = presidents[list(presidents.keys())[st.session_state.current_president_index]]['image']
        st.image(image, width=300)
    with col3:
        st.write(' ')


    col1 = st.columns([1])[0]
    with col1:
        # Display the current president's name
        st.write("")
        st.write(f"<h2 style='font-weight: bold;text-align:center'>{list(presidents.keys())[st.session_state.current_president_index]}</h2>", unsafe_allow_html=True)
        # Display the current president's biography text
        biography = presidents[list(presidents.keys())[st.session_state.current_president_index]]['biography']
        st.markdown(f"<div style='text-align:center'>{biography}</div>", unsafe_allow_html=True)





    # Embed a JavaScript code to scroll to the top of the page when the button is clicked
    js_scroll_to_top = """
    <script>
    document.querySelector("button").addEventListener("click", function() {
        window.scrollTo({top: 0, behavior: 'smooth'});
    });
    </script>
    """
    st.components.v1.html(js_scroll_to_top)
    
if __name__ == '__main__':
    show_explore_page()