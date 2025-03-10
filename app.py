# Import required libraries
from dotenv import load_dotenv
import streamlit as st
from streamlit_extras import add_vertical_space as avs
import os
import PyPDF2
from groq import Groq
from auth import render_auth_page, init_auth_state
import sqlite3
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
import json
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
import re

# Import database functions
from database import init_db, save_application, get_all_applications, update_application_status, get_user_profile, save_user_profile, create_tables, save_job_description

# Load environment variables and initialize client
load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Constants
MIXTRAL_MODEL = "mixtral-8x7b-32768"
ATS_PROMPT = """As an experienced ATS (Applicant Tracking System), analyze the following resume against the job description. Provide:
1. A percentage match with the Job Role
2. Missing keywords
3. A profile summary

Resume: {text}
Job Description: {jd}"""

# Replace the HTML warning with Streamlit styling
#def display_violation_warning():
    #st.warning("⚠️ Violations may affect your assessment evaluation.")

# Initialize session state
def init_session_state():
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'selected_role' not in st.session_state:
        st.session_state.selected_role = None

# AI Response Functions
def get_ai_response(messages, temperature=0.7, max_tokens=2048):
    try:
        response = client.chat.completions.create(
            messages=messages,
            model=MIXTRAL_MODEL,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)}"

def get_chat_response(user_input):
    # Initialize the Groq client
    client = Groq(api_key=st.secrets["GROQ_API_KEY"])
    
    # Initialize the messages list with the system prompt
    messages = [
        {"role": "system", "content": "You are a helpful career advisor, providing guidance on job applications, interview preparation, and career development."},
        {"role": "user", "content": user_input}
    ]
    
    try:
        response = client.chat.completions.create(
            model="mixtral-8x7b-32768",  # Groq's model
            messages=messages,
            max_tokens=500,
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return "I apologize, but I'm having trouble connecting to the service right now. Please try again later."

# Utility Functions
def parse_pdf(uploaded_file):
    reader = PyPDF2.PdfReader(uploaded_file)
    return ' '.join(page.extract_text() for page in reader.pages)

def save_resume_file(uploaded_file, user_id):
    if uploaded_file is None:
        return None
    
    # Create directory if it doesn't exist
    os.makedirs('uploads/resumes', exist_ok=True)
    
    # Save file with user_id in name
    file_extension = uploaded_file.name.split('.')[-1]
    file_path = f'uploads/resumes/resume_{user_id}.{file_extension}'
    
    with open(file_path, 'wb') as f:
        f.write(uploaded_file.getbuffer())
    
    return file_path

# UI Components
def render_role_selection():
    st.title("AI-powered profile analysis and Evaluation System")
    col1, col2, col3 = st.columns(3)
    
    roles = {"Admin": col1, "HR": col2, "Candidate": col3}
    for role, col in roles.items():
        with col:
            if st.button(role, use_container_width=True):
                st.session_state.selected_role = role.lower()
                st.rerun()

def render_logout_button():
    if st.sidebar.button("Logout"):
        st.session_state.is_authenticated = False
        st.session_state.current_user = None
        st.session_state.selected_role = None
        st.rerun()

def render_resume_analysis():
    col1, col2 = st.columns([3, 2])
    with col1:
        st.title("AI-powered Profile Analysis")
        st.header("Navigate the Job Market with Confidence!")
        st.markdown("""<p style='text-align: justify;'>
                    Introducing AI-powered Profile Analysis, an ATS-Optimized Resume Analyzer your ultimate solution for optimizing job applications and accelerating career growth.</p>""", 
                    unsafe_allow_html=True)
    
    with col2:
        st.image('https://cdn.dribbble.com/userupload/12500996/file/original-b458fe398a6d7f4e9999ce66ec856ff9.gif', 
                use_container_width=True)
    
    avs.add_vertical_space(0)
    
    col1, col2 = st.columns([3,2])
    with col1:
        st.markdown("<h2 style='text-align: center;'>Resume Analysis</h2>", unsafe_allow_html=True)
        jd = st.text_area("Paste the Job Role", height=150)
        uploaded_file = st.file_uploader("Upload Your Resume", type="pdf", help="Please upload the pdf")
        
        if st.button("Analyze Resume") and uploaded_file and jd:
            text = parse_pdf(uploaded_file)
            messages = [{"role": "user", "content": ATS_PROMPT.format(text=text, jd=jd)}]
            response = get_ai_response(messages)
            st.markdown("### Analysis Results")
            st.write(response)
            
            # Save application to database
            save_application(
                st.session_state.current_user['id'],
                jd,
                text
            )
            st.success("Application saved successfully!")
            st.session_state.job_description_saved = True

        # New button to save the job description
        if st.button("Save Job Role"):
            if jd:
                if save_job_description(jd):
                    st.success("Job Role saved successfully!")
                    st.session_state.job_description_saved = True
                else:
                    st.error("Failed to save job description.")
            else:
                st.warning("Please enter a job description.")

def render_career_chat():
    st.title("Career Advisor Chat")
    st.header("Get personalized career advice and support!")
    
    # Add a warning about API key requirement
    if "GROQ_API_KEY" not in st.secrets:
        st.warning("Please set up your Groq API key in the app secrets to use the chat feature.")
        st.stop()
    
    # Initialize chat history in session state if it doesn't exist
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Display chat history
    for index, message in enumerate(st.session_state.chat_history):
        with st.chat_message("user"):
            st.write(message[0])
            # Add delete button for user messages
            if st.button("Delete", key=f"delete_user_{index}"):
                st.session_state.chat_history.pop(index)
                st.experimental_rerun()  # Refresh the app to update the chat history
        
        with st.chat_message("assistant"):
            st.write(message[1])
            # Add delete button for assistant messages
            if st.button("Delete", key=f"delete_assistant_{index}"):
                st.session_state.chat_history.pop(index)
                st.experimental_rerun()  # Refresh the app to update the chat history
    
    # Get user input
    user_input = st.chat_input("Ask me anything about your career...")
     
    if user_input:
        # Display user message
        with st.chat_message("user"):
            st.write(user_input)
        
        # Get and display assistant response
        with st.chat_message("assistant"):
            response = get_chat_response(user_input)
            st.write(response)
        
        # Add to chat history
        st.session_state.chat_history.append((user_input, response))

def render_dashboard(role):
    st.title(f"{role.upper()} Dashboard")
    st.image('https://cdn.dribbble.com/users/1787323/screenshots/11307210/media/ec074b669c542464034a6e84d63a8145.gif', 
             use_column_width=True)
    st.write(f"{role.capitalize()} functionality coming soon...")

# Add HR dashboard function
def render_hr_dashboard():
    st.title("HR Dashboard")
    
    # Add tabs for different views
    tab1, tab2 = st.tabs(["Applications", "Candidate Profiles"])
    
    with tab1:
        render_hr_applications()  # Your existing applications view
    
    with tab2:
        if 'viewing_candidate' in st.session_state:
            if st.button("← Back to Candidate List"):
                del st.session_state.viewing_candidate
                st.rerun()
            render_candidate_view(st.session_state.viewing_candidate, "hr")
        else:
            render_candidate_list()

def render_admin_dashboard():
    st.title("Admin Dashboard")
    
    # Add tabs for different views
    tab1, tab2, tab3 = st.tabs(["Overview", "Applications", "Candidate Profiles"])
    
    with tab1:
        render_admin_overview()  # Your existing overview
    
    with tab2:
        render_admin_applications()  # Your existing applications view
    
    with tab3:
        if 'viewing_candidate' in st.session_state:
            if st.button("← Back to Candidate List"):
                del st.session_state.viewing_candidate
                st.rerun()
            render_candidate_view(st.session_state.viewing_candidate, "admin")
        else:
            render_candidate_list()

def get_candidate_details(user_id):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    
    # Get profile and application statistics
    c.execute('''
        SELECT 
            p.*,
            COUNT(DISTINCT a.id) as total_applications,
            SUM(CASE WHEN a.status = 'pending' THEN 1 ELSE 0 END) as pending_count,
            SUM(CASE WHEN a.status = 'shortlisted' THEN 1 ELSE 0 END) as shortlisted_count,
            SUM(CASE WHEN a.status = 'rejected' THEN 1 ELSE 0 END) as rejected_count
        FROM user_profiles p
        LEFT JOIN applications a ON p.user_id = a.user_id
        WHERE p.user_id = ?
        GROUP BY p.id
    ''', (user_id,))
    
    result = c.fetchone()
    
    # Get recent applications
    c.execute('''
        SELECT id, job_description, status, created_at
        FROM applications
        WHERE user_id = ?
        ORDER BY created_at DESC
        LIMIT 5
    ''', (user_id,))
    
    recent_applications = c.fetchall()
    conn.close()
    
    return result, recent_applications

def delete_user_and_data(user_id):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    try:
        # Start transaction
        c.execute('BEGIN TRANSACTION')
        
        # Get user profile to delete resume file if exists
        c.execute('SELECT resume_path FROM user_profiles WHERE user_id = ?', (user_id,))
        profile = c.fetchone()
        if profile and profile[0]:
            try:
                os.remove(profile[0])  # Delete resume file
            except OSError:
                pass  # File might not exist
        
        # Delete related records in order
        c.execute('DELETE FROM notifications WHERE user_id = ?', (user_id,))
        c.execute('DELETE FROM applications WHERE user_id = ?', (user_id,))
        c.execute('DELETE FROM user_profiles WHERE user_id = ?', (user_id,))
        c.execute('DELETE FROM users WHERE id = ?', (user_id,))
        
        # Commit transaction
        conn.commit()
        return True
    except Exception as e:
        print(f"Error deleting user: {str(e)}")
        conn.rollback()
        return False
    finally:
        conn.close()

def render_candidate_view(user_id, user_role="hr"):
    profile, applications = get_candidate_details(user_id)
    
    if not profile:
        st.warning("Candidate profile not found.")
        return
    
    # Display profile information
    st.title(f"Candidate Profile: {profile[2]}")  # full_name
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Personal Information")
        st.write(f"**Email:** {profile[3]}")
        st.write(f"**Phone:** {profile[4]}")
        st.write(f"**Location:** {profile[5]}")
        if profile[6]:  # LinkedIn URL
            st.write(f"**LinkedIn:** [{profile[6]}]({profile[6]})")
        
        # Skills
        if profile[7]:
            st.subheader("Skills")
            skills_list = profile[7].split('\n')
            cols = st.columns(3)
            for i, skill in enumerate(skills_list):
                if skill.strip():
                    cols[i % 3].write(f"- {skill.strip()}")
        
        # Experience
        if profile[8]:
            st.subheader("Work Experience")
            st.write(profile[8])
        
        # Education
        if profile[9]:
            st.subheader("Education")
            st.write(profile[9])
    
    with col2:
        st.subheader("Application History")
        for app in applications:
            with st.expander(f"Application #{app[0]} - {app[3].split()[0]}"):
                st.write(f"**Status:** {app[2].upper()}")
                st.write("**Job Role Preview:**")
                st.text_area(
                    "", 
                    app[1][:200] + "..." if len(app[1]) > 200 else app[1],
                    height=100,
                    disabled=True,
                    key=f"jd_preview_{app[0]}_{user_id}"
                )
        
        # Add download resume button
        resume_path = download_resume(user_id)
        if resume_path:
            with open(resume_path, "rb") as f:
                st.download_button(
                    label="Download Resume",
                    data=f,
                    file_name=f"resume_{user_id}.pdf",
                    mime="application/pdf"
                )
        else:
            st.warning("No resume found for this candidate.")

def render_candidate_list():
    st.subheader("Candidate Directory")
    
    # Add bulk delete option for admin
    if st.session_state.current_user['role'] == 'admin':
        selected_candidates = []  # To store selected candidates for deletion
        
        if st.button("🗑️ Delete Selected", type="primary"):
            st.warning("⚠️ Are you sure you want to delete the selected profiles? This action cannot be undone!")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("✔️ Yes, Delete", type="primary"):
                    deleted_count = 0
                    for user_id in st.session_state.get('selected_candidates', []):
                        if delete_user_and_data(user_id):
                            deleted_count += 1
                    if deleted_count > 0:
                        st.success(f"{deleted_count} profile(s) deleted successfully!")
                        st.session_state.selected_candidates = []
                        time.sleep(1)
                        st.rerun()
            with col2:
                if st.button("❌ Cancel"):
                    st.rerun()
    
    # Get filtered candidates
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    
    query = '''
        SELECT DISTINCT
            u.id,
            p.full_name,
            p.location,
            COUNT(a.id) as application_count,
            MAX(a.created_at) as last_application,
            p.email,
            p.phone
        FROM users u
        LEFT JOIN user_profiles p ON u.id = p.user_id
        LEFT JOIN applications a ON u.id = a.user_id
        WHERE u.role = 'candidate'
        GROUP BY u.id
        ORDER BY p.full_name
    '''
    
    c.execute(query)
    candidates = c.fetchall()
    conn.close()
    
    # Initialize selected candidates in session state if not exists
    if 'selected_candidates' not in st.session_state:
        st.session_state.selected_candidates = []
    
    # Display candidates
    for candidate in candidates:
        with st.expander(f"{candidate[1] or 'Unnamed'} - {candidate[3]} applications"):
            cols = st.columns([0.5, 2, 1, 1])
            
            # Add checkbox for selection (admin only)
            if st.session_state.current_user['role'] == 'admin':
                with cols[0]:
                    if st.checkbox("", key=f"select_{candidate[0]}", 
                                 value=candidate[0] in st.session_state.selected_candidates):
                        if candidate[0] not in st.session_state.selected_candidates:
                            st.session_state.selected_candidates.append(candidate[0])
                    else:
                        if candidate[0] in st.session_state.selected_candidates:
                            st.session_state.selected_candidates.remove(candidate[0])
            
            with cols[1]:
                st.write(f"**Location:** {candidate[2] or 'Not specified'}")
                st.write(f"**Email:** {candidate[5] or 'Not specified'}")
                st.write(f"**Phone:** {candidate[6] or 'Not specified'}")
            
            with cols[2]:
                st.write(f"**Applications:** {candidate[3]}")
                if candidate[4]:
                    st.write(f"**Last Application:** {candidate[4].split()[0]}")
            
            with cols[3]:
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("View", key=f"view_{candidate[0]}"):
                        st.session_state.viewing_candidate = candidate[0]
                        st.rerun()
                with col2:
                    if st.session_state.current_user['role'] == 'admin':
                        if st.button("Delete", key=f"delete_{candidate[0]}", type="primary"):
                            if delete_user_and_data(candidate[0]):
                                st.success("Profile deleted successfully!")
                                time.sleep(1)
                                st.rerun()
                            else:
                                st.error("Failed to delete profile")

def get_application_stats():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    
    # Get overall statistics
    c.execute('''
        SELECT 
            COUNT(*) as total_applications,
            SUM(CASE WHEN status = 'pending' THEN 1 ELSE 0 END) as pending,
            SUM(CASE WHEN status = 'shortlisted' THEN 1 ELSE 0 END) as shortlisted,
            SUM(CASE WHEN status = 'rejected' THEN 1 ELSE 0 END) as rejected
        FROM applications
    ''')
    stats = c.fetchone()
    
    # Get statistics by time period
    c.execute('''
        SELECT 
            COUNT(*) as total,
            SUM(CASE WHEN status = 'pending' THEN 1 ELSE 0 END) as pending,
            SUM(CASE WHEN status = 'shortlisted' THEN 1 ELSE 0 END) as shortlisted,
            SUM(CASE WHEN status = 'rejected' THEN 1 ELSE 0 END) as rejected
        FROM applications
        WHERE created_at >= date('now', '-30 days')
    ''')
    monthly_stats = c.fetchone()
    
    conn.close()
    return stats, monthly_stats

def render_admin_overview():
    st.subheader("Application Statistics")
    
    # Get statistics
    overall_stats, monthly_stats = get_application_stats()
    
    # Display overall statistics
    st.markdown("### Overall Statistics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Applications",
            overall_stats[0],
            f"+{monthly_stats[0]} in 30 days"
        )
    
    with col2:
        st.metric(
            "Pending Review",
            overall_stats[1],
            f"+{monthly_stats[1]} in 30 days",
            delta_color="off"
        )
    
    with col3:
        st.metric(
            "Shortlisted",
            overall_stats[2],
            f"+{monthly_stats[2]} in 30 days",
            delta_color="inverse"
        )
    
    with col4:
        st.metric(
            "Rejected",
            overall_stats[3],
            f"+{monthly_stats[3]} in 30 days",
            delta_color="inverse"
        )
    
    # Visual representation
    st.markdown("### Application Status Distribution")
    
    # Prepare data for pie chart
    labels = ['Pending', 'Shortlisted', 'Rejected']
    sizes = [overall_stats[1], overall_stats[2], overall_stats[3]]

    # Create figure with custom colors
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=sizes,
        hole=.3,
        marker=dict(colors=['#FFA500', '#32CD32', '#DC143C'])
    )])
    
    fig.update_layout(
        showlegend=True,
        width=800,
        height=400,
        margin=dict(t=0, b=0, l=0, r=0)
    )
    
    st.plotly_chart(fig)
    
    # Monthly trend
    st.markdown("### Monthly Application Trend")
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    
    c.execute('''
        SELECT 
            date(created_at) as date,
            COUNT(*) as count,
            SUM(CASE WHEN status = 'pending' THEN 1 ELSE 0 END) as pending,
            SUM(CASE WHEN status = 'shortlisted' THEN 1 ELSE 0 END) as shortlisted,
            SUM(CASE WHEN status = 'rejected' THEN 1 ELSE 0 END) as rejected
        FROM applications
        WHERE created_at >= date('now', '-30 days')
        GROUP BY date(created_at)
        ORDER BY date
    ''')
    
    daily_stats = c.fetchall()
    conn.close()
    
    if daily_stats:
        dates = [stat[0] for stat in daily_stats]
        total = [stat[1] for stat in daily_stats]
        pending = [stat[2] for stat in daily_stats]
        shortlisted = [stat[3] for stat in daily_stats]
        rejected = [stat[4] for stat in daily_stats]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=dates, y=total,
            name='Total',
            line=dict(color='#1f77b4', width=2)
        ))
        
        fig.add_trace(go.Scatter(
            x=dates, y=pending,
            name='Pending',
            line=dict(color='#FFA500', width=2)
        ))
        
        fig.add_trace(go.Scatter(
            x=dates, y=shortlisted,
            name='Shortlisted',
            line=dict(color='#32CD32', width=2)
        ))
        
        fig.add_trace(go.Scatter(
            x=dates, y=rejected,
            name='Rejected',
            line=dict(color='#DC143C', width=2)
        ))
        
        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Number of Applications",
            hovermode='x unified',
            width=800,
            height=400,
            margin=dict(t=0, b=0, l=0, r=0)
        )
        
        st.plotly_chart(fig)
    else:
        st.info("No application data available for the past 30 days")

def render_admin_applications():
    st.subheader("Application Management")
    
    # Filter options
    col1, col2, col3 = st.columns(3)
    with col1:
        status_filter = st.selectbox(
            "Filter by Status",
            ["All", "Pending", "Shortlisted", "Reviewed", "Rejected"]
        )
    with col2:
        date_filter = st.selectbox(
            "Time Period",
            ["All Time", "Last 7 Days", "Last 30 Days", "Last 90 Days"]
        )
    with col3:
        name_filter = st.text_input(
            "Search by Name",
            placeholder="Enter candidate name..."
        )
    
    # Build query based on filters
    query = '''
        SELECT 
            a.id,
            u.username,
            p.full_name,
            a.job_description,
            a.status,
            a.created_at,
            t.status as test_status,
            t.score as test_score,
            t.violations_count,
            u.id as user_id
        FROM applications a
        JOIN users u ON a.user_id = u.id
        LEFT JOIN user_profiles p ON u.id = p.user_id
        LEFT JOIN tests t ON a.id = t.application_id
    '''
    
    params = []
    where_clauses = []
    
    if status_filter != "All":
        where_clauses.append("a.status = ?")
        params.append(status_filter.lower())
    
    if date_filter != "All Time":
        days = {
            "Last 7 Days": 7,
            "Last 30 Days": 30,
            "Last 90 Days": 90
        }
        where_clauses.append("a.created_at >= date('now', ?)")
        params.append(f'-{days[date_filter]} days')
    
    if name_filter:
        where_clauses.append("(LOWER(p.full_name) LIKE ? OR LOWER(u.username) LIKE ?)")
        search_term = f"%{name_filter.lower()}%"
        params.extend([search_term, search_term])
    
    if where_clauses:
        query += " WHERE " + " AND ".join(where_clauses)
    
    query += " ORDER BY a.created_at DESC"
    
    # Execute query
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute(query, params)
    applications = c.fetchall()
    conn.close()
    
    # Display applications
    if not applications:
        st.info("No applications found matching the criteria.")
        return
        
    for app in applications:
        with st.expander(f"Application #{app[0]} - {app[2] or app[1]}"):
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.write("**Job Role:**")
                st.text_area(
                    "",
                    app[3][:200] + "...",
                    height=100,
                    disabled=True,
                    key=f"admin_jd_{app[0]}"
                )
            
            with col2:
                st.write(f"**Status:** {app[4].upper()}")
                st.write(f"**Submitted:** {app[5]}")
                
                # Add Assessment Status section for shortlisted applications
                if app[4] == 'shortlisted':
                    st.markdown("### Assessment Status")
                    if app[6]:  # test_status exists
                        if app[6] == 'completed':
                            st.markdown(f"""
                                <div style='
                                    padding: 20px; 
                                    border-radius: 10px; 
                                    margin: 10px 0;
                                    background-color: #000000;
                                    border-left: 5px solid #28a745;
                                '>
                                    <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;'>
                                        <h4 style='margin: 0;'>Assessment Details</h4>
                                        <span style='
                                            background-color: #28a745; 
                                            color: white; 
                                            padding: 5px 10px; 
                                            border-radius: 15px;
                                            font-size: 0.9em;
                                        '>
                                            Completed
                                        </span>
                                    </div>
                                    <p style='margin: 5px 0;'><strong>Score:</strong> {app[7]}%</p>
                                    <p style='margin: 5px 0;'>
                                        <strong style='color: red;'>Violations:</strong> {app[8] or 0}
                                    </p>
                                </div>
                            """, unsafe_allow_html=True)
                            
                        else:
                            st.markdown("""
                                <div style='
                                    padding: 20px; 
                                    border-radius: 10px; 
                                    margin: 10px 0;
                                    background-color: #f0f2f6;
                                    border-left: 5px solid #ffc107;
                                '>
                                    <div style='display: flex; justify-content: space-between; align-items: center;'>
                                        <h4 style='margin: 0;'>Assessment Details</h4>
                                        <span style='
                                            background-color: #ffc107; 
                                            color: white; 
                                            padding: 5px 10px; 
                                            border-radius: 15px;
                                            font-size: 0.9em;
                                        '>
                                            Pending
                                        </span>
                                    </div>
                                    <p style='margin-top: 10px;'>Assessment is pending completion by the candidate.</p>
                                </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.markdown("""
                            <div style='
                                padding: 20px; 
                                border-radius: 10px; 
                                margin: 10px 0;
                                background-color: #000000;
                                border-left: 5px solid #dc3545;
                            '>
                                <div style='display: flex; justify-content: space-between; align-items: center;'>
                                    <h4 style='margin: 0;'>Assessment Details</h4>
                                    <span style='
                                        background-color: #dc3545; 
                                        color: white; 
                                        padding: 5px 10px; 
                                        border-radius: 15px;
                                        font-size: 0.9em;
                                    '>
                                        Not Started
                                    </span>
                                </div>
                                <p style='margin-top: 10px;'>Candidate has not started the assessment yet.</p>
                
                            </div>
                        """, unsafe_allow_html=True)
                
                # Status update controls
                new_status = st.selectbox(
                    "Update Status",
                    ["pending", "reviewed", "shortlisted", "rejected"],
                    index=["pending", "reviewed", "shortlisted", "rejected"].index(app[4]),
                    key=f"admin_status_{app[0]}"
                )
                
                if st.button("Update Status", key=f"admin_update_{app[0]}"):
                    if update_application_status(app[0], new_status):
                        st.success("Status updated successfully!")
                        time.sleep(1)
                        st.rerun()
                
                # View full profile button
                if st.button("View Full Profile", key=f"admin_profile_{app[0]}"):
                    st.session_state.viewing_candidate = app[8]  # user_id
                    st.rerun()

            # Add Notes Section after the existing columns
            st.markdown("""
                <div style='
                    margin-top: 20px;
                    padding-top: 20px;
                    border-top: 1px solid #555;
                '>
                    <h4 style='color: white;'>📝 Notes</h4>
                </div>
            """, unsafe_allow_html=True)
            
            # Get existing notes
            conn = sqlite3.connect('users.db')
            c = conn.cursor()
            c.execute('''
                SELECT n.id, n.note_text, n.created_at, u.username
                FROM application_notes n
                JOIN users u ON n.hr_id = u.id
                WHERE n.application_id = ?
                ORDER BY n.created_at DESC
            ''', (app[0],))
            notes = c.fetchall()
            
            # Display existing notes
            if notes:
                for note in notes:
                    st.markdown(f"""
                        <div style='
                            padding: 15px;
                            margin: 10px 0;
                            background-color: #1a1a1a;
                            border-radius: 5px;
                            border-left: 3px solid #1E88E5;
                        '>
                            <p style='color: #888; margin: 0; font-size: 0.8em;'>
                                By {note[3]} on {note[2]}
                            </p>
                            <p style='color: white; margin: 5px 0;'>{note[1]}</p>
                        </div>
                    """, unsafe_allow_html=True)
            
            # Add new note
            new_note = st.text_area(
                "Add a note",
                key=f"note_input_{app[0]}",
                height=100,
                placeholder="Enter your notes about this application..."
            )
            
            if st.button("Add Note", key=f"add_note_{app[0]}"):
                if new_note.strip():
                    try:
                        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        c.execute('''
                            INSERT INTO application_notes 
                            (application_id, hr_id, note_text, created_at)
                            VALUES (?, ?, ?, ?)
                        ''', (
                            app[0],
                            st.session_state.current_user['id'],
                            new_note,
                            current_time
                        ))
                        conn.commit()
                        st.success("Note added successfully!")
                        time.sleep(1)
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error adding note: {str(e)}")
                else:
                    st.warning("Please enter a note before adding.")
            
            conn.close()
            
            # Add delete button at the bottom of col2
            with col2:
                st.markdown("---")  # Add separator
                if st.button("🗑️ Delete Application", key=f"delete_app_{app[0]}", type="primary"):
                    st.warning("⚠️ Are you sure you want to delete this application? This action cannot be undone!")
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("✔️ Yes, Delete", key=f"confirm_delete_{app[0]}", type="primary"):
                            if delete_application(app[0]):
                                st.success("Application deleted successfully!")
                                time.sleep(1)
                                st.rerun()
                            else:
                                st.error("Failed to delete application")
                    with col2:
                        if st.button("❌ Cancel", key=f"cancel_delete_{app[0]}"):
                            st.rerun()

def delete_application(app_id):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    try:
        # Start transaction
        c.execute('BEGIN TRANSACTION')
        
        # Delete related records in order
        c.execute('DELETE FROM test_results WHERE test_id IN (SELECT id FROM tests WHERE application_id = ?)', (app_id,))
        c.execute('DELETE FROM tests WHERE application_id = ?', (app_id,))
        c.execute('DELETE FROM application_notes WHERE application_id = ?', (app_id,))
        c.execute('DELETE FROM applications WHERE id = ?', (app_id,))
        
        # Commit transaction
        conn.commit()
        return True
    except Exception as e:
        print(f"Error deleting application: {str(e)}")
        conn.rollback()
        return False
    finally:
        conn.close()

def render_hr_applications():
    st.subheader("Application Management")
    
    # Filter and search options
    col1, col2, col3 = st.columns(3)
    
    with col1:
        status_filter = st.selectbox(
            "Filter by Status",
            ["All", "Pending", "Shortlisted", "Reviewed", "Rejected"]
        )
    
    with col2:
        date_filter = st.selectbox(
            "Time Period",
            ["All Time", "Today", "Last 7 Days", "Last 30 Days"]
        )
    
    with col3:
        search_term = st.text_input("Search by candidate name")
    
    # Build query based on filters
    query = '''
        SELECT 
            a.id,
            u.username,
            p.full_name,
            a.job_description,
            a.resume_text,
            a.status,
            a.created_at,
            u.id as user_id,
            t.status as test_status,
            COALESCE(t.score, 0) as test_score,
            COALESCE(t.violations_count, 0) as violations_count,
            t.completed_at,
            t.id as test_id
        FROM applications a
        JOIN users u ON a.user_id = u.id
        LEFT JOIN user_profiles p ON u.id = p.user_id
        LEFT JOIN tests t ON a.id = t.application_id
        WHERE 1=1
    '''
    
    params = []
    
    if status_filter != "All":
        query += " AND a.status = ?"
        params.append(status_filter.lower())
    
    if date_filter != "All Time":
        if date_filter == "Today":
            query += " AND date(a.created_at) = date('now')"
        else:
            days = {"Last 7 Days": 7, "Last 30 Days": 30}
            query += " AND a.created_at >= date('now', ?)"
            params.append(f'-{days[date_filter]} days')
    
    if search_term:
        query += " AND (LOWER(p.full_name) LIKE ? OR LOWER(u.username) LIKE ?)"
        search_pattern = f"%{search_term.lower()}%"
        params.extend([search_pattern, search_pattern])
    
    query += " ORDER BY a.created_at DESC"
    
    # Execute query
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute(query, params)
    applications = c.fetchall()
    conn.close()
    
    # Display applications
    if not applications:
        st.info("No applications found matching the criteria.")
        return
    
    for idx, app in enumerate(applications, 1):  # Add index to enumeration
        with st.expander(f"Application #{app[0]} - {app[2] or app[1]}"):
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.write("**Job Role:**")
                st.text_area("", app[3], height=100, key=f"jd_{app[0]}", disabled=True)
                
                st.write("**Resume Text:**")
                st.text_area("", app[4], height=100, key=f"resume_{app[0]}", disabled=True)
            
            with col2:
                st.write(f"**Current Status:** {app[5].upper()}")
                st.write(f"**Submitted:** {app[6]}")
                
                # Add Assessment Status section with updated details
                if app[5] == 'shortlisted':
                    st.markdown("### Assessment Status")
                    
                    # Get test data
                    test_id = app[12]  # test_id
                    test_status = app[8]  # test_status
                    test_score = app[9]  # test_score
                    violations = app[10]  # violations_count
                    completed_at = app[11]  # completed_at
                    
                    # Query the latest test data directly from the database
                    conn = sqlite3.connect('users.db')
                    c = conn.cursor()
                    c.execute('''
                        SELECT status, score, violations_count, completed_at
                        FROM tests
                        WHERE application_id = ?
                        ORDER BY created_at DESC
                        LIMIT 1
                    ''', (app[0],))
                    latest_test = c.fetchone()
                    conn.close()
                    
                    if latest_test:
                        test_status = latest_test[0]
                        test_score = latest_test[1]
                        violations = latest_test[2]
                        completed_at = latest_test[3]
                    
                    # Format the score
                    try:
                        score_display = f"{float(test_score):.1f}%" if test_score is not None else "N/A"
                    except (ValueError, TypeError):
                        score_display = "N/A"
                    
                    # Format violations
                    try:
                        violations = int(violations)
                    except (ValueError, TypeError):
                        violations = 0
                    
                    if test_status == 'completed':
                        st.markdown(f"""
                            <div style='
                                padding: 20px; 
                                border-radius: 10px; 
                                margin: 10px 0;
                                background-color: #000000;
                                border-left: 5px solid #28a745;
                            '>
                                <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;'>
                                    <h4 style='margin: 0; color: white;'>Assessment #{idx} Details</h4>
                                    <span style='
                                        background-color: #28a745; 
                                        color: white; 
                                        padding: 5px 10px; 
                                        border-radius: 15px;
                                        font-size: 0.9em;
                                    '>
                                        Completed
                                    </span>
                                </div>
                                <p style='margin: 5px 0; color: white;'><strong>Score:</strong> {score_display}</p>
                                <p style='margin: 5px 0; color: white;'>
                                    <strong style='color: #ff4444;'>Violations:</strong> {violations}
                                </p>
                            </div>
                        """, unsafe_allow_html=True)
                        
                      
                            
                    elif test_status == 'in_progress':
                        st.markdown(f"""
                            <div style='
                                padding: 20px; 
                                border-radius: 10px; 
                                margin: 10px 0;
                                background-color: #000000;
                                border-left: 5px solid #ffc107;
                            '>
                                <div style='display: flex; justify-content: space-between; align-items: center;'>
                                    <h4 style='margin: 0; color: white;'>Assessment #{idx} Details</h4>
                                    <span style='
                                        background-color: #ffc107; 
                                        color: white; 
                                        padding: 5px 10px; 
                                        border-radius: 15px;
                                        font-size: 0.9em;
                                    '>
                                        In Progress
                                    </span>
                                </div>
                                <p style='margin-top: 10px; color: white;'>Your assessment is currently in progress.</p>
                                <a href="/test?id={app[0]}" target="_blank">
                                    <button style='
                                        background-color: #1E88E5;
                                        color: white;
                                        padding: 10px 20px;
                                        border: none;
                                        border-radius: 5px;
                                        cursor: pointer;
                                        margin-top: 10px;
                                        width: 100%;
                                    '>
                                        Continue Assessment
                                    </button>
                                </a>
                            </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                            <div style='
                                padding: 20px; 
                                border-radius: 10px; 
                                margin: 10px 0;
                                background-color: #000000;
                                border-left: 5px solid #dc3545;
                            '>
                                <div style='display: flex; justify-content: space-between; align-items: center;'>
                                    <h4 style='margin: 0; color: white;'>Assessment #{idx} Details</h4>
                                    <span style='
                                        background-color: #dc3545; 
                                        color: white; 
                                        padding: 5px 10px; 
                                        border-radius: 15px;
                                        font-size: 0.9em;
                                    '>
                                        Not Started
                                    </span>
                                </div>
                                <p style='margin-top: 10px; color: white;'>Candidate has not started the assessment yet.</p>
                            </div>
                        """, unsafe_allow_html=True)
                
                # Status update controls
                new_status = st.selectbox(
                    "Update Status",
                    ["pending", "reviewed", "shortlisted", "rejected"],
                    index=["pending", "reviewed", "shortlisted", "rejected"].index(app[5]),
                    key=f"status_{app[0]}"
                )
                
                if st.button("Update Status", key=f"update_{app[0]}"):
                    if update_application_status(app[0], new_status):
                        st.success("Status updated successfully!")
                        time.sleep(1)
                        st.rerun()
                
                # View full profile button
                if st.button("View Full Profile", key=f"profile_{app[0]}"):
                    st.session_state.viewing_candidate = app[7]  # user_id
                    st.rerun()

            # Add Notes Section after the existing columns
            st.markdown("""
                <div style='
                    margin-top: 20px;
                    padding-top: 20px;
                    border-top: 1px solid #555;
                '>
                    <h4 style='color: white;'>📝 Notes</h4>
                </div>
            """, unsafe_allow_html=True)
            
            # Get existing notes
            conn = sqlite3.connect('users.db')
            c = conn.cursor()
            c.execute('''
                SELECT n.id, n.note_text, n.created_at, u.username
                FROM application_notes n
                JOIN users u ON n.hr_id = u.id
                WHERE n.application_id = ?
                ORDER BY n.created_at DESC
            ''', (app[0],))
            notes = c.fetchall()
            
            # Display existing notes
            if notes:
                for note in notes:
                    st.markdown(f"""
                        <div style='
                            padding: 15px;
                            margin: 10px 0;
                            background-color: #1a1a1a;
                            border-radius: 5px;
                            border-left: 3px solid #1E88E5;
                        '>
                            <p style='color: #888; margin: 0; font-size: 0.8em;'>
                                By {note[3]} on {note[2]}
                            </p>
                            <p style='color: white; margin: 5px 0;'>{note[1]}</p>
                        </div>
                    """, unsafe_allow_html=True)
            
            # Add new note
            new_note = st.text_area(
                "Add a note",
                key=f"note_input_{app[0]}",
                height=100,
                placeholder="Enter your notes about this application..."
            )
            
            if st.button("Add Note", key=f"add_note_{app[0]}"):
                if new_note.strip():
                    try:
                        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        c.execute('''
                            INSERT INTO application_notes 
                            (application_id, hr_id, note_text, created_at)
                            VALUES (?, ?, ?, ?)
                        ''', (
                            app[0],
                            st.session_state.current_user['id'],
                            new_note,
                            current_time
                        ))
                        
                        # Add notification for other HR members
                        c.execute('''
                            INSERT INTO notifications 
                            (user_id, message, created_at, is_read)
                            SELECT 
                                u.id,
                                ?,
                                ?,
                                0
                            FROM users u
                            WHERE u.role = 'hr' AND u.id != ?
                        ''', (
                            f"New note added to Application #{app[0]}",
                            current_time,
                            st.session_state.current_user['id']
                        ))
                        
                        conn.commit()
                        st.success("Note added successfully!")
                        time.sleep(1)
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error adding note: {str(e)}")
                else:
                    st.warning("Please enter a note before adding.")
            
            conn.close()

def get_candidate_notes(candidate_id):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    try:
        c.execute('''
            SELECT n.id, n.hr_id, n.note_text, n.created_at
            FROM candidate_notes n
            WHERE n.candidate_id = ?
            ORDER BY n.created_at DESC
        ''', (candidate_id,))
        return c.fetchall()
    except Exception as e:
        print(f"Error getting notes: {str(e)}")
        return []
    finally:
        conn.close()

def save_candidate_note(candidate_id, note_text, hr_id):
    if not note_text.strip():
        return False
        
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    try:
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        c.execute('''
            INSERT INTO candidate_notes 
            (candidate_id, hr_id, note_text, created_at)
            VALUES (?, ?, ?, ?)
        ''', (candidate_id, hr_id, note_text, current_time))
        
        # Add notification for candidate
        c.execute('''
            INSERT INTO notifications 
            (user_id, message, created_at, is_read)
            VALUES (?, ?, ?, ?)
        ''', (
            candidate_id,
            "HR has added a new note to your profile.",
            current_time,
            False
        ))
        
        conn.commit()
        return True
    except Exception as e:
        print(f"Error saving note: {str(e)}")
        conn.rollback()
        return False
    finally:
        conn.close()

def analyze_candidate_profile(resume_text, test_score, violations):
    """
    Analyze candidate profile using Random Forest and Groq AI
    """
    analysis = {
        'strengths': [],
        'weaknesses': [],
        'improvements': []
    }
    
    # Extract skills and create feature vector
    skills_pattern = r'python|java|javascript|react|node|sql|machine learning|ai|docker|aws|cloud|testing|agile|scrum'
    found_skills = re.findall(skills_pattern, resume_text.lower())
    skill_count = Counter(found_skills)
    
    # Create feature vector for Random Forest
    features = {
        'test_score': float(test_score) if test_score is not None else 0,
        'violations': int(violations) if violations is not None else 0,
        'skill_count': len(found_skills),
        'unique_skills': len(set(found_skills))
    }
    
    # Use Groq AI for detailed analysis
    analysis_prompt = f"""
    As an AI career advisor, analyze this candidate profile and provide specific recommendations:
    
    Technical Assessment:
    - Test Score: {features['test_score']}%
    - Violations: {features['violations']}
    
    Skills Profile:
    {', '.join(found_skills)}
    
    Resume Text:
    {resume_text}
    
    Please provide:
    1. Key strengths
    2. Areas needing improvement
    3. Specific action items for career development
    4. Industry-relevant skill recommendations
    """
    
    try:
        # Get AI analysis using existing Groq client
        messages = [{"role": "user", "content": analysis_prompt}]
        ai_response = get_ai_response(messages)
        
        # Parse AI response
        sections = ai_response.split('\n\n')
        for section in sections:
            if 'strength' in section.lower():
                analysis['strengths'].extend([s.strip() for s in section.split('\n')[1:] if s.strip()])
            elif 'improvement' in section.lower():
                analysis['weaknesses'].extend([s.strip() for s in section.split('\n')[1:] if s.strip()])
            elif 'action' in section.lower():
                analysis['improvements'].extend([s.strip() for s in section.split('\n')[1:] if s.strip()])
    
    except Exception as e:
        st.error(f"Error in AI analysis: {str(e)}")
        # Fallback to basic analysis
        if features['test_score'] >= 80:
            analysis['strengths'].append("Strong technical knowledge demonstrated in assessment")
        elif features['test_score'] >= 60:
            analysis['strengths'].append("Satisfactory technical foundation")
            analysis['improvements'].append("Focus on improving technical depth")
        else:
            analysis['weaknesses'].append("Technical knowledge needs significant improvement")
            analysis['improvements'].append("Recommended to review core technical concepts")
    
    # Random Forest based skill gap analysis
    try:
        # Create a simple dataset for the Random Forest
        X = pd.DataFrame([features])
        
        # Define simple thresholds based on industry standards
        skill_thresholds = {
            'technical': 75,
            'professional': 70,
            'skill_diversity': 5
        }
        
        # Train a simple Random Forest model
        rf = RandomForestClassifier(n_estimators=10, random_state=42)
        
        # Generate synthetic training data based on thresholds
        X_train = pd.DataFrame([
            {'test_score': 90, 'violations': 0, 'skill_count': 8, 'unique_skills': 6},
            {'test_score': 70, 'violations': 1, 'skill_count': 5, 'unique_skills': 4},
            {'test_score': 50, 'violations': 3, 'skill_count': 3, 'unique_skills': 2}
        ])
        y_train = ['excellent', 'good', 'needs_improvement']
        
        rf.fit(X_train, y_train)
        
        # Get prediction
        prediction = rf.predict(X)[0]
        
        # Add Random Forest based recommendations
        if prediction == 'needs_improvement':
            analysis['improvements'].extend([
                "Consider focusing on core technical skills",
                "Work on reducing test violations",
                "Expand your skill set breadth"
            ])
        elif prediction == 'good':
            analysis['improvements'].extend([
                "Consider advanced certifications",
                "Focus on specialized skills in your domain"
            ])
        else:
            analysis['strengths'].append("Well-rounded technical profile")
            
    except Exception as e:
        st.error(f"Error in Random Forest analysis: {str(e)}")
    
    return analysis

def render_candidate_profile():
    st.title("My Profile")
    
    # Get existing profile
    profile = get_user_profile(st.session_state.current_user['id'])
    
    tab1, tab2, tab3, tab4 = st.tabs(["Profile Details", "Applications Status", "Technical Assessment", "Detailed Report"])
    
    with tab1:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Personal Information")
            
            # Create form for profile information
            with st.form("profile_form"):
                full_name = st.text_input("Full Name", value=profile[2] if profile else "")
                email = st.text_input("Email Address", value=profile[3] if profile else "")
                phone = st.text_input("Phone Number", value=profile[4] if profile else "")
                location = st.text_input("Location", value=profile[5] if profile else "")
                linkedin_url = st.text_input("LinkedIn Profile URL", value=profile[6] if profile else "")
                
                st.subheader("Professional Details")
                skills = st.text_area("Skills (one per line)", value=profile[7] if profile else "")
                experience = st.text_area("Work Experience", value=profile[8] if profile else "")
                education = st.text_area("Education", value=profile[9] if profile else "")
                
                uploaded_file = st.file_uploader("Upload Resume (PDF)", type="pdf")
                
                if st.form_submit_button("Save Profile"):
                    resume_path = profile[10] if profile else None
                    if uploaded_file:
                        resume_path = save_resume_file(uploaded_file, st.session_state.current_user['id'])
                    
                    profile_data = (full_name, email, phone, location, linkedin_url,
                                  skills, experience, education, resume_path)
                    save_user_profile(st.session_state.current_user['id'], profile_data)
                    st.success("Profile updated successfully!")
                    st.rerun()
    
    with tab2:
        st.subheader("📋 My Applications")
        
        conn = sqlite3.connect('users.db')
        c = conn.cursor()
        
        # First check if the violations_count column exists
        c.execute("PRAGMA table_info(tests)")
        columns = [col[1] for col in c.fetchall()]
        
        # Build the query based on whether violations_count exists
        if 'violations_count' in columns:
            query = '''
                SELECT 
                    a.id,
                    a.job_description,
                    a.status,
                    a.created_at,
                    t.score,
                    COALESCE(t.violations_count, 0) as violations_count,
                    CASE 
                        WHEN a.status = 'shortlisted' THEN '🎉 Congratulations! You have been shortlisted for Technical Assessment!'
                        WHEN a.status = 'reviewed' THEN '👀 Your application has been reviewed'
                        WHEN a.status = 'rejected' THEN '❌ Application not selected'
                        ELSE '⏳ Under review'
                    END as status_message
                FROM applications a
                LEFT JOIN tests t ON a.id = t.application_id
                WHERE a.user_id = ?
                ORDER BY a.created_at DESC
            '''
        else:
            # Fallback query without violations_count
            query = '''
                SELECT 
                    a.id,
                    a.job_description,
                    a.status,
                    a.created_at,
                    t.score,
                    0 as violations_count,
                    CASE 
                        WHEN a.status = 'shortlisted' THEN '🎉 Congratulations! You have been shortlisted for Technical Assessment!'
                        WHEN a.status = 'reviewed' THEN '👀 Your application has been reviewed'
                        WHEN a.status = 'rejected' THEN '❌ Application not selected'
                        ELSE '⏳ Under review'
                    END as status_message
                FROM applications a
                LEFT JOIN tests t ON a.id = t.application_id
                WHERE a.user_id = ?
                ORDER BY a.created_at DESC
            '''
        
        c.execute(query, (st.session_state.current_user['id'],))
        applications = c.fetchall()
        
        if applications:
            for app in applications:
                with st.expander(f"Application #{app[0]} - {app[2].upper()}", expanded=True):
                    st.info(app[6])  # Status message
                    st.caption(f"Submitted on: {app[3]}")
                    
                    # Display test results if available
                    if app[4] is not None:  # If test score exists
                        st.markdown("#### Test Results")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Score", f"{app[4]}%")
                        with col2:
                            violations = int(app[5] or 0)  # Convert to integer and handle None
                            st.metric(
                                "Violations Detected",
                                violations,
                                delta=None,
                                delta_color="inverse" if violations > 0 else "off"
                            )
                            
                            
                    
                    st.text_area(
                        "Job Role", 
                        value=app[1], 
                        height=100, 
                        disabled=True,
                        key=f"app_jd_{app[0]}"
                    )
        else:
            st.info("You haven't submitted any applications yet.")
        
        conn.close()
    
    with tab3:
        st.markdown("""
            <div style='text-align: center; padding: 20px;'>
                <h2>🎯 Technical Skills Assessment</h2>
            </div>
        """, unsafe_allow_html=True)
        
        # Get assessment details including test status
        conn = sqlite3.connect('users.db')
        c = conn.cursor()
        
        c.execute('''
            SELECT 
                a.id as app_id,
                t.id as test_id,
                t.status as test_status,
                COALESCE(t.score, 0) as score,
                t.completed_at,
                a.status as application_status,
                COALESCE(t.violations_count, 0) as violations_count
            FROM applications a
            LEFT JOIN tests t ON a.id = t.application_id
            WHERE a.user_id = ? AND a.status = 'shortlisted'
            ORDER BY a.created_at DESC
        ''', (st.session_state.current_user['id'],))
        
        assessments = c.fetchall()
        conn.close()
        
        if assessments:
            for index, assessment in enumerate(assessments, 1):
                with st.container():
                    app_id, test_id, test_status, score, completed_at, app_status, violations = assessment
                    
                    # Convert score to float and format it
                    try:
                        score = float(score)
                        score_display = f"{score:.1f}%"
                    except (ValueError, TypeError):
                        score_display = "N/A"
                    
                    # Convert violations to int
                    try:
                        violations = int(violations)
                    except (ValueError, TypeError):
                        violations = 0
                    
                    if test_status == 'completed':
                        st.markdown(f"""
                            <div style='
                                padding: 20px; 
                                border-radius: 10px; 
                                margin: 10px 0;
                                background-color: #000000;
                                border-left: 5px solid #28a745;
                            '>
                                <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;'>
                                    <h4 style='margin: 0; color: white;'>Assessment #{index} Details</h4>
                                    <span style='
                                        background-color: #28a745; 
                                        color: white; 
                                        padding: 5px 10px; 
                                        border-radius: 15px;
                                        font-size: 0.9em;
                                    '>
                                        Completed
                                    </span>
                                </div>
                                <p style='margin: 5px 0; color: white;'><strong>Score:</strong> {score_display}</p>
                                <p style='margin: 5px 0; color: white;'>
                                    <strong style='color: #ff4444;'>Violations:</strong> {violations}
                                </p>
                            </div>
                        """, unsafe_allow_html=True)
                        
                        if violations > 0:
                            st.warning("⚠️ Violations may affect your assessment evaluation.")
                            
                    elif test_status == 'in_progress':
                        st.markdown(f"""
                            <div style='
                                padding: 20px; 
                                border-radius: 10px; 
                                margin: 10px 0;
                                background-color: #000000;
                                border-left: 5px solid #ffc107;
                            '>
                                <div style='display: flex; justify-content: space-between; align-items: center;'>
                                    <h4 style='margin: 0; color: white;'>Assessment #{index} Details</h4>
                                    <span style='
                                        background-color: #ffc107; 
                                        color: white; 
                                        padding: 5px 10px; 
                                        border-radius: 15px;
                                        font-size: 0.9em;
                                    '>
                                        In Progress
                                    </span>
                                </div>
                                <p style='margin-top: 10px; color: white;'>Your assessment is currently in progress.</p>
                                <a href="/test?id={app_id}" target="_blank">
                                    <button style='
                                        background-color: #1E88E5;
                                        color: white;
                                        padding: 10px 20px;
                                        border: none;
                                        border-radius: 5px;
                                        cursor: pointer;
                                        margin-top: 10px;
                                        width: 100%;
                                    '>
                                        Continue Assessment
                                    </button>
                                </a>
                            </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                            <div style='
                                padding: 20px; 
                                border-radius: 10px; 
                                margin: 10px 0;
                                background-color: #000000;
                                border-left: 5px solid #dc3545;
                            '>
                                <div style='display: flex; justify-content: space-between; align-items: center;'>
                                    <h4 style='margin: 0; color: white;'>Assessment #{index} Details</h4>
                                    <span style='
                                        background-color: #dc3545; 
                                        color: white; 
                                        padding: 5px 10px; 
                                        border-radius: 15px;
                                        font-size: 0.9em;
                                    '>
                                        Not Started
                                    </span>
                                </div>
                                <p style='margin-top: 10px; color: white;'>You have not started your assessment yet.</p>
                                <div style='margin-top: 15px; padding: 15px; background-color: #1a1a1a; border-radius: 5px;'>
                                    <h5 style='color: white; margin: 0 0 10px 0;'>Assessment Overview</h5>
                                    <ul style='color: white; margin: 0; padding-left: 20px;'>
                                        <li>Duration: 10 minutes</li>
                                        <li>Questions: Multiple Choice</li>
                                        <li>Webcam Required</li>
                                    </ul>
                                </div>
                                <a href="/test?id={app_id}" target="_blank">
                                    <button style='
                                        background-color: #1E88E5;
                                        color: white;
                                        padding: 10px 20px;
                                        border: none;
                                        border-radius: 5px;
                                        cursor: pointer;
                                        margin-top: 15px;
                                        width: 100%;
                                        font-weight: bold;
                                    '>
                                        Start Assessment
                                    </button>
                                </a>
                            </div>
                        """, unsafe_allow_html=True)
        else:
            st.info("No assessments available. You will see assessments here when your applications are shortlisted.")

    with tab4:
        st.markdown("""
            <div style='text-align: center; padding: 20px;'>
                <h2>🎯 Here is your Detailed report </h2>
            </div>
        """, unsafe_allow_html=True)
        
        # Get all completed assessments for the candidate
        conn = sqlite3.connect('users.db')
        c = conn.cursor()
        
        c.execute('''
            SELECT 
                t.id,
                t.score,
                t.violations_count,
                t.completed_at,
                a.id as app_id
            FROM applications a
            LEFT JOIN tests t ON a.id = t.application_id
            WHERE a.user_id = ? 
            AND t.status = 'completed'
            ORDER BY t.completed_at DESC
        ''', (st.session_state.current_user['id'],))
        
        completed_assessments = c.fetchall()
        
        # Create assessment selection dropdown
        if completed_assessments:
            assessment_options = {
                f"Assessment #{i+1} (Score: {float(test[1]):.1f}%, Date: {test[3].split()[0]})": test[0]
                for i, test in enumerate(completed_assessments)
            }
            assessment_options["All Assessments"] = "all"
            
            selected_assessment = st.selectbox(
                "Choose Assessment for Report",
                options=list(assessment_options.keys()),
                key="career_analysis_assessment"
            )
            
            # Get profile data
            c.execute('''
                SELECT skills, experience, education
                FROM user_profiles
                WHERE user_id = ?
            ''', (st.session_state.current_user['id'],))
            
            profile_data = c.fetchone()
            
            if profile_data:
                skills, experience, education = profile_data
                profile_text = f"{skills or ''} {experience or ''} {education or ''}"
                
                if selected_assessment == "All Assessments":
                    # Aggregate scores and violations for overall analysis
                    avg_score = sum(float(test[1]) for test in completed_assessments) / len(completed_assessments)
                    total_violations = sum(int(test[2] or 0) for test in completed_assessments)
                    analysis = analyze_candidate_profile(profile_text, avg_score, total_violations)
                    st.info("📊 Showing analysis based on all completed assessments")
                else:
                    # Get selected assessment data
                    test_id = assessment_options[selected_assessment]
                    selected_test = next(
                        (test for test in completed_assessments if test[0] == test_id),
                        None
                    )
                    if selected_test:
                        analysis = analyze_candidate_profile(
                            profile_text,
                            float(selected_test[1]),
                            int(selected_test[2] or 0)
                        )
                        st.info(f"📊 Showing Report for {selected_assessment}")
                
                # Display analysis results
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("""
                        <div style='
                            padding: 20px;
                            border-radius: 10px;
                            margin: 10px 0;
                            background-color: #000000;
                            border-left: 5px solid #28a745;
                        '>
                            <h4 style='color: white; margin-bottom: 15px;'>💪 Strengths</h4>
                    """, unsafe_allow_html=True)
                    
                    for strength in analysis['strengths']:
                        st.markdown(f"<p style='color: white; margin: 5px 0;'>• {strength}</p>", unsafe_allow_html=True)
                    
                    st.markdown("</div>", unsafe_allow_html=True)
                
                with col2:
                    st.markdown("""
                        <div style='
                            padding: 20px;
                            border-radius: 10px;
                            margin: 10px 0;
                            background-color: #000000;
                            border-left: 5px solid #dc3545;
                        '>
                            <h4 style='color: white; margin-bottom: 15px;'>🎯 Areas for Improvement</h4>
                    """, unsafe_allow_html=True)
                    
                    for weakness in analysis['weaknesses']:
                        st.markdown(f"<p style='color: white; margin: 5px 0;'>• {weakness}</p>", unsafe_allow_html=True)
                    
                    st.markdown("</div>", unsafe_allow_html=True)
                
                # Recommendations section
                st.markdown("""
                    <div style='
                        padding: 20px;
                        border-radius: 10px;
                        margin: 20px 0;
                        background-color: #000000;
                        border-left: 5px solid #1e88e5;
                    '>
                        <h4 style='color: white; margin-bottom: 15px;'>📚 Recommended Improvements</h4>
                """, unsafe_allow_html=True)
                
                for improvement in analysis['improvements']:
                    st.markdown(f"<p style='color: white; margin: 5px 0;'>• {improvement}</p>", unsafe_allow_html=True)
                
                st.markdown("</div>", unsafe_allow_html=True)
                
                # Add career resources
                st.markdown("""
                    <div style='
                        padding: 20px;
                        border-radius: 10px;
                        margin: 20px 0;
                        background-color: #000000;
                        border-left: 5px solid #ffc107;
                    '>
                        <h4 style='color: white; margin-bottom: 15px;'>🔍 Recommended Resources</h4>
                        <ul style='color: white; margin: 10px 0;'>
                            <li>Online Learning Platforms (Coursera, Udemy)</li>
                            <li>Technical Documentation and Tutorials</li>
                            <li>Industry Certifications</li>
                            <li>Professional Networking Events</li>
                        </ul>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.warning("Please complete your profile to get a detailed career analysis.")
        else:
            st.warning("Complete at least one technical assessment to get a detailed career analysis.")
        
        conn.close()

def update_application_status(app_id, new_status):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    
    try:
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Update application status
        c.execute('''
            UPDATE applications 
            SET status = ?
            WHERE id = ?
        ''', (new_status, app_id))
        
        # Get user_id for notification
        c.execute('SELECT user_id FROM applications WHERE id = ?', (app_id,))
        user_id = c.fetchone()[0]
        
        # Create notification based on status
        notification_message = {
            'shortlisted': "🎉 Congratulations! Your application has been shortlisted for Assessment.",
            'rejected': "We regret to inform you that your application was not selected.",
            'reviewed': "Your application has been reviewed by our team.",
            'pending': "Your application is under review."
        }
        
        # Insert notification
        c.execute('''
            INSERT INTO notifications 
            (user_id, message, created_at, is_read)
            VALUES (?, ?, ?, ?)
        ''', (
            user_id,
            notification_message.get(new_status, "Your application status has been updated."),
            current_time,
            False
        ))
        
        conn.commit()
        return True
        
    except Exception as e:
        print(f"Error updating status: {e}")
        conn.rollback()
        return False
    finally:
        conn.close()

def get_user_notifications(user_id):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    
    c.execute('''
        SELECT id, message, created_at, is_read
        FROM notifications
        WHERE user_id = ?
        ORDER BY created_at DESC
    ''', (user_id,))
    
    notifications = c.fetchall()
    conn.close()
    return notifications

def mark_notification_as_read(notification_id):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    
    try:
        c.execute('''
            UPDATE notifications
            SET is_read = TRUE
            WHERE id = ?
        ''', (notification_id,))
        conn.commit()
        return True
    except:
        conn.rollback()
        return False
    finally:
        conn.close()

def render_candidate_applications():
    st.subheader("My Applications")
    
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    
    c.execute('''
        SELECT 
            a.id,
            a.job_description,
            a.status,
            a.created_at,
            CASE 
                WHEN a.status = 'shortlisted' THEN '🎉 Congratulations! Your application has been shortlisted for Assessment!'
                WHEN a.status = 'reviewed' THEN '👀 Your application has been reviewed'
                WHEN a.status = 'rejected' THEN '❌ Application not selected'
                WHEN a.status = 'pending' THEN '⏳ Under review'
                ELSE '👀 Application being reviewed'
            END as status_message
        FROM applications a
        WHERE a.user_id = ?
        ORDER BY a.created_at DESC
    ''', (st.session_state.current_user['id'],))
    
    applications = c.fetchall()
    
    if not applications:
        st.info("You haven't submitted any applications yet.")
        return
    
    for app in applications:
        with st.expander(f"Application from {app[3].split()[0]} - {app[2].upper()}", expanded=True):
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown("#### Job Description")
                st.text_area("", app[1], height=150, key=f"jd_{app[0]}", disabled=True)
            
            with col2:
                st.markdown("#### Status")
                st.info(app[4])  # Display status message
                
                # Show test link only for shortlisted applications
                if app[2] == 'shortlisted':
                    # Check if test exists
                    c.execute('SELECT id, completed_at, score FROM tests WHERE application_id = ?', (app[0],))
                    test = c.fetchone()
                    
                    if test:
                        if test[1]:  # If test is completed
                            st.success(f"Test completed! Score: {test[2]}%")
                        else:
                            st.warning("You have a pending skills assessment test!")
                            st.markdown(f'''
                                <div style="
                                    background-color: #f0f2f6;
                                    border-radius: 10px;
                                    padding: 20px;
                                    margin: 10px 0;
                                    text-align: center;
                                ">
                                    <h4 style="margin-bottom: 10px;">Skills Assessment Test</h4>
                                    <p style="color: #666; margin-bottom: 15px;">⏱️ Duration: 30 minutes</p>
                                    <button onclick="window.location.href='test?id={test[0]}'" style="
                                        background-color: #1E88E5;
                                        color: white;
                                        padding: 12px 24px;
                                        border: none;
                                        border-radius: 4px;
                                        cursor: pointer;
                                        font-size: 16px;
                                        width: 100%;
                                        margin-top: 10px;
                                        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                                        transition: all 0.3s ease;
                                    " onmouseover="this.style.backgroundColor='#1976D2'"
                                      onmouseout="this.style.backgroundColor='#1E88E5'">
                                        Take Skills Assessment Test
                                    </button>
                                </div>
                            ''', unsafe_allow_html=True)
                    else:
                        # Generate new test
                        st.markdown(f'''
                            <div style="
                                background-color: #f0f2f6;
                                border-radius: 10px;
                                padding: 20px;
                                margin: 10px 0;
                                text-align: center;
                            ">
                                <h4 style="margin-bottom: 10px;">Skills Assessment Test</h4>
                                <p style="color: #666; margin-bottom: 15px;">⏱️ Duration: 30 minutes</p>
                                <p style="color: #666; margin-bottom: 15px;">A test will be generated based on your profile</p>
                                <button onclick="window.location.href='test?id={app[0]}'" style="
                                    background-color: #1E88E5;
                                    color: white;
                                    padding: 12px 24px;
                                    border: none;
                                    border-radius: 4px;
                                    cursor: pointer;
                                    font-size: 16px;
                                    width: 100%;
                                    margin-top: 10px;
                                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                                    transition: all 0.3s ease;
                                " onmouseover="this.style.backgroundColor='#1976D2'"
                                  onmouseout="this.style.backgroundColor='#1E88E5'">
                                    Start Skills Assessment Test
                                </button>
                            </div>
                        ''', unsafe_allow_html=True)
    
    # Add video preview section with webcam test
    st.markdown("""
        <div style="margin: 20px 0;">
            <h4>📹 Video Monitoring Requirements</h4>
            <p>This test requires webcam monitoring. Please test your camera before starting.</p>
        </div>
    """, unsafe_allow_html=True)
    
    if st.button("Test Webcam"):
        st.session_state.show_webcam = True
    
    if st.session_state.get('show_webcam', False):
        st.markdown("""
            <div style="margin: 10px 0;">
                <video id="webcamPreview" autoplay playsinline style="width: 100%; max-width: 400px; border-radius: 8px;"></video>
            </div>
            <script>
                async function initWebcam() {
                    try {
                        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
                        const video = document.getElementById('webcamPreview');
                        video.srcObject = stream;
                    } catch (err) {
                        alert('Error accessing webcam: ' + err.message);
                    }
                }
                initWebcam();
            </script>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("✅ Camera Working"):
                st.session_state.camera_verified = True
                st.session_state.show_webcam = False
                st.success("Camera verified successfully!")
                st.rerun()
        with col2:
            if st.button("❌ Having Issues"):
                st.error("Please ensure your camera is working before starting the test.")
                st.markdown("""
                    Common troubleshooting steps:
                    1. Allow camera access in your browser
                    2. Check if another application is using the camera
                    3. Restart your browser
                    4. Update your browser to the latest version
                """)

# Add new function for video monitoring during test
def setup_video_monitoring():
    st.markdown("""
        <div style="position: fixed; top: 20px; right: 20px; width: 200px; z-index: 1000;">
            <video id="monitoringVideo" autoplay playsinline 
                style="width: 100%; border-radius: 8px; border: 2px solid #1E88E5;">
            </video>
        </div>
        <script>
            async function setupMonitoring() {
                try {
                    const stream = await navigator.mediaDevices.getUserMedia({ 
                        video: { 
                            width: 320, 
                            height: 240,
                            facingMode: "user"
                        } 
                    });
                    const video = document.getElementById('monitoringVideo');
                    video.srcObject = stream;
                    
                    // Optional: Record video
                    const mediaRecorder = new MediaRecorder(stream);
                    const chunks = [];
                    
                    mediaRecorder.ondataavailable = (e) => {
                        if (e.data.size > 0) {
                            chunks.push(e.data);
                        }
                    };
                    
                    mediaRecorder.onstop = () => {
                        const blob = new Blob(chunks, { type: 'video/webm' });
                        // Here you can implement the upload logic
                        console.log('Recording stopped, blob created');
                    };
                    
                    // Start recording
                    mediaRecorder.start(1000); // Capture in 1-second chunks
                    
                    // Store recorder reference for later use
                    window.testMediaRecorder = mediaRecorder;
                    
                } catch (err) {
                    console.error('Error in video monitoring:', err);
                    alert('Error setting up video monitoring: ' + err.message);
                }
            }
            
            // Initialize monitoring when page loads
            setupMonitoring();
            
            // Cleanup when page unloads
            window.onbeforeunload = () => {
                if (window.testMediaRecorder && window.testMediaRecorder.state === 'recording') {
                    window.testMediaRecorder.stop();
                }
            };
        </script>
    """, unsafe_allow_html=True)

def render_test_page():
    # Verify camera access first
    if not st.session_state.get('camera_verified', False):
        st.warning("⚠️ Please verify your camera is working before starting the test")
        render_camera_check()
        return
    
    # Setup video monitoring
    setup_video_monitoring()
    
    # Rest of your test page code...
    st.title("Skills Assessment Test")
    # ... existing test page content ...

def render_camera_check():
    st.markdown("""
        <div style="text-align: center; margin: 20px 0;">
            <h3>📹 Camera Check Required</h3>
            <p>This test requires webcam monitoring. Please verify your camera is working.</p>
        </div>
    """, unsafe_allow_html=True)
    
    if st.button("Start Camera Check"):
        st.session_state.show_webcam = True
    
    if st.session_state.get('show_webcam', False):
        st.markdown("""
            <div style="display: flex; justify-content: center; margin: 20px 0;">
                <video id="webcamPreview" autoplay playsinline 
                    style="width: 100%; max-width: 400px; border-radius: 8px; border: 2px solid #ddd;">
                </video>
            </div>
            <script>
                async function initWebcam() {
                    try {
                        const stream = await navigator.mediaDevices.getUserMedia({ 
                            video: { facingMode: "user" } 
                        });
                        const video = document.getElementById('webcamPreview');
                        video.srcObject = stream;
                    } catch (err) {
                        alert('Error accessing webcam: ' + err.message);
                    }
                }
                initWebcam();
            </script>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("✅ Camera Working"):
                st.session_state.camera_verified = True
                st.session_state.show_webcam = False
                st.success("Camera verified successfully! You can now start the test.")
                time.sleep(2)
                st.rerun()
        with col2:
            if st.button("❌ Having Issues"):
                st.error("Please ensure your camera is working before starting the test.")
                st.markdown("""
                    Troubleshooting steps:
                    1. Allow camera access in your browser settings
                    2. Check if another application is using the camera
                    3. Restart your browser
                    4. Make sure your camera is properly connected
                    5. Update your browser to the latest version
                """)

def create_tables():
    """Create database tables if they don't exist"""
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    
    # Create tables with updated schema
    c.execute('''
        CREATE TABLE IF NOT EXISTS tests (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            application_id INTEGER,
            status TEXT DEFAULT 'pending',
            score REAL,
            completed_at DATETIME,
            violations_count INTEGER DEFAULT 0,
            FOREIGN KEY (application_id) REFERENCES applications (id)
        )
    ''')
    
    c.execute('''
        CREATE TABLE IF NOT EXISTS test_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            test_id INTEGER,
            answers TEXT,
            score REAL,
            completed_at DATETIME,
            violations_count INTEGER DEFAULT 0,
            FOREIGN KEY (test_id) REFERENCES tests (id)
        )
    ''')
    
    # Add violations_count column to existing tables if it doesn't exist
    try:
        c.execute('ALTER TABLE tests ADD COLUMN violations_count INTEGER DEFAULT 0')
    except sqlite3.OperationalError:
        # Column already exists
        pass
        
    try:
        c.execute('ALTER TABLE test_results ADD COLUMN violations_count INTEGER DEFAULT 0')
    except sqlite3.OperationalError:
        # Column already exists
        pass
    
    # Add application_notes table
    c.execute('''
        CREATE TABLE IF NOT EXISTS application_notes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            application_id INTEGER NOT NULL,
            hr_id INTEGER NOT NULL,
            note_text TEXT NOT NULL,
            created_at DATETIME NOT NULL,
            FOREIGN KEY (application_id) REFERENCES applications (id),
            FOREIGN KEY (hr_id) REFERENCES users (id)
        )
    ''')
    
    conn.commit()
    conn.close()

def init_db():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    
    # Create users table
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  username TEXT UNIQUE NOT NULL,
                  password TEXT NOT NULL,
                  role TEXT NOT NULL)''')
    
    # Create applications table
    c.execute('''CREATE TABLE IF NOT EXISTS applications
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  user_id INTEGER NOT NULL,
                  job_description TEXT NOT NULL,
                  resume_text TEXT NOT NULL,
                  status TEXT DEFAULT 'pending',
                  created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                  FOREIGN KEY (user_id) REFERENCES users (id))''')
    
    # Create user_profiles table
    c.execute('''CREATE TABLE IF NOT EXISTS user_profiles
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  user_id INTEGER UNIQUE NOT NULL,
                  full_name TEXT,
                  email TEXT,
                  phone TEXT,
                  location TEXT,
                  linkedin_url TEXT,
                  skills TEXT,
                  experience TEXT,
                  education TEXT,
                  resume_path TEXT,
                  FOREIGN KEY (user_id) REFERENCES users (id))''')
    
    # Create tests table
    c.execute('''CREATE TABLE IF NOT EXISTS tests
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  application_id INTEGER NOT NULL,
                  status TEXT DEFAULT 'not_started',
                  score REAL,
                  violations_count INTEGER DEFAULT 0,
                  created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                  completed_at DATETIME,
                  FOREIGN KEY (application_id) REFERENCES applications (id))''')
    
    # Create notifications table
    c.execute('''CREATE TABLE IF NOT EXISTS notifications
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  user_id INTEGER NOT NULL,
                  message TEXT NOT NULL,
                  created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                  is_read BOOLEAN DEFAULT 0,
                  FOREIGN KEY (user_id) REFERENCES users (id))''')
    
    # Create job_descriptions table
    c.execute('''CREATE TABLE IF NOT EXISTS job_descriptions
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  description TEXT NOT NULL,
                  created_at DATETIME DEFAULT CURRENT_TIMESTAMP)''')
    
    # Create application_notes table
    c.execute('''CREATE TABLE IF NOT EXISTS application_notes
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  application_id INTEGER NOT NULL,
                  hr_id INTEGER NOT NULL,
                  note_text TEXT NOT NULL,
                  created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                  FOREIGN KEY (application_id) REFERENCES applications (id),
                  FOREIGN KEY (hr_id) REFERENCES users (id))''')
    
    conn.commit()
    conn.close()

def get_db_connection():
    """Helper function to get database connection"""
    try:
        conn = sqlite3.connect('data/users.db')
        conn.row_factory = sqlite3.Row
        return conn
    except Exception as e:
        st.error(f"Database connection error: {str(e)}")
        return None

def download_resume(user_id):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    
    # Get the resume path for the user
    c.execute('SELECT resume_path FROM user_profiles WHERE user_id = ?', (user_id,))
    resume_path = c.fetchone()
    
    conn.close()
    
    if resume_path and resume_path[0]:
        return resume_path[0]  # Return the path to the resume file
    return None

def main():
    try:
        st.set_page_config(page_title="AI-powered Profile Analysis", layout="wide")
        
        # Initialize database first
        init_db()
        
        # Initialize session states
        init_session_state()
        init_auth_state()
        
        if not st.session_state.is_authenticated:
            render_auth_page()
        else:
            render_logout_button()
            if st.session_state.selected_role == "candidate":
                tab1, tab2, tab3 = st.tabs(["Resume Analysis", "Career Chat", "My Profile"])
                with tab1:
                    render_resume_analysis()
                with tab2:
                    render_career_chat()
                with tab3:
                    render_candidate_profile()
            elif st.session_state.selected_role == "hr":
                render_hr_dashboard()
            elif st.session_state.selected_role == "admin":
                render_admin_dashboard()
    except Exception as e:
        st.error(f"An error occurred in main: {str(e)}")
        print(f"Error details: {str(e)}")

if __name__ == "__main__":
    main()