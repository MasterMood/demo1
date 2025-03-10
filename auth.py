import streamlit as st
import sqlite3
import hashlib
from datetime import datetime

def init_auth_state():
    if 'is_authenticated' not in st.session_state:
        st.session_state.is_authenticated = False
    if 'current_user' not in st.session_state:
        st.session_state.current_user = None

def init_db():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  username TEXT UNIQUE NOT NULL,
                  password TEXT NOT NULL,
                  role TEXT NOT NULL,
                  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
    conn.commit()
    conn.close()

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def authenticate_user(username, password):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('SELECT * FROM users WHERE username=? AND password=?', 
              (username, hash_password(password)))
    user = c.fetchone()
    conn.close()
    return user

def create_user(username, password, role):
    try:
        conn = sqlite3.connect('users.db')
        c = conn.cursor()
        c.execute('INSERT INTO users (username, password, role) VALUES (?, ?, ?)',
                 (username, hash_password(password), role))
        conn.commit()
        conn.close()
        return True
    except sqlite3.IntegrityError:
        return False

def render_login():
    st.markdown("""
        <style>
        .main-img {
            max-height: 300px;
            width: auto;
            margin: 20px 0;
        }
        .small-img {
            max-height: 120px;
            width: auto;
            margin: 10px 0;
        }
        </style>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("<h3>Login to AI-powered profile analysis</h3>", unsafe_allow_html=True)

        # Main login image with fixed height
        st.markdown("""
            <div style='display: flex; justify-content: center;'>
                <img src='https://img.freepik.com/free-vector/recruit-agent-analyzing-candidates_74855-4565.jpg' 
                     class='main-img'>
            </div>
        """, unsafe_allow_html=True)
        
        # Additional images in a row with fixed height
        st.markdown("<div style='margin-top: 20px;'>", unsafe_allow_html=True)
        img_col1, img_col2, img_col3 = st.columns(3)
        with img_col1:
            st.markdown("""
                <div style='display: flex; justify-content: center;'>
                    <img src='' 
                         class='small-img'>
                </div>
            """, unsafe_allow_html=True)
        with img_col2:
            st.markdown("""
                <div style='display: flex; justify-content: center;'>
                    <img src='' 
                         class='small-img'>
                </div>
            """, unsafe_allow_html=True)
        with img_col3:
            st.markdown("""
                <div style='display: flex; justify-content: center;'>
                    <img src='' 
                         class='small-img'>
                </div>
            """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<br>" * 2, unsafe_allow_html=True)  # Reduced vertical space
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Login", use_container_width=True)
            
            if submitted:
                if username and password:
                    user = authenticate_user(username, password)
                    if user:
                        st.session_state.is_authenticated = True
                        st.session_state.current_user = {
                            'id': user[0],
                            'username': user[1],
                            'role': user[3]
                        }
                        st.session_state.selected_role = user[3].lower()
                        st.success("Login successful!")
                        st.rerun()
                    else:
                        st.error("Invalid username or password")
                else:
                    st.error("Please fill in all fields")

def render_signup():
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("<h3>Sign Up for AI-powered profile analysis</h3>", unsafe_allow_html=True)
       
        # Main signup image with fixed height
        st.markdown("""
    <div style='display: flex; justify-content: center;'>
        <img src='https://img.freepik.com/free-vector/sign-up-concept-illustration_114360-7965.jpg' 
             class='main-img' width='450' height='450'>
    </div>
""", unsafe_allow_html=True)

        
        # Additional images in a row with fixed height
        st.markdown("<div style='margin-top: 20px;'>", unsafe_allow_html=True)
        img_col1, img_col2, img_col3 = st.columns(3)
        with img_col1:
            st.markdown("""
                <div style='display: flex; justify-content: center;'>
                    <img src='' 
                         class='small-img'>
                </div>
            """, unsafe_allow_html=True)
        with img_col2:
            st.markdown("""
                <div style='display: flex; justify-content: center;'>
                    <img src='' 
                         class='small-img'>
                </div>
            """, unsafe_allow_html=True)
        with img_col3:
            st.markdown("""
                <div style='display: flex; justify-content: center;'>
                    <img src='' 
                         class='small-img'>
                </div>
            """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<br>" * 2, unsafe_allow_html=True)
        with st.form("signup_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            confirm_password = st.text_input("Confirm Password", type="password")
            role = st.selectbox("Role", ["candidate", "hr", "admin"])
            submitted = st.form_submit_button("Sign Up", use_container_width=True)
            
            if submitted:
                if username and password and confirm_password:
                    if password != confirm_password:
                        st.error("Passwords do not match")
                    elif len(password) < 6:
                        st.error("Password must be at least 6 characters long")
                    else:
                        if create_user(username, password, role):
                            st.success("Account created successfully! Please login.")
                            st.session_state.show_login = True
                            st.rerun()
                        else:
                            st.error("Username already exists")
                else:
                    st.error("Please fill in all fields")

def render_auth_page():
    init_auth_state()
    init_db()
    
    if 'show_login' not in st.session_state:
        st.session_state.show_login = True
    
    # Add a header with logo/banner
    st.markdown("""
        <div style='text-align: center; padding: 1rem;'>
            <h1 style='color: #ffffff;'>Welcome to AI-powered profile analysis</h1>
        </div>
    """, unsafe_allow_html=True)
    
    # Main authentication container
    if st.session_state.show_login:
        render_login()
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("Don't have an account? Sign Up", use_container_width=True):
                st.session_state.show_login = False
                st.rerun()
    else:
        render_signup()
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("Already have an account? Login", use_container_width=True):
                st.session_state.show_login = True
                st.rerun()
    
    
    