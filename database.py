import sqlite3
from datetime import datetime

def init_db():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    
    # Create users table if not exists
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  username TEXT UNIQUE NOT NULL,
                  password TEXT NOT NULL,
                  role TEXT NOT NULL,
                  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
    
    # Create user_profiles table if not exists
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
                  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                  FOREIGN KEY (user_id) REFERENCES users(id))''')
    
    # Create applications table if not exists
    c.execute('''CREATE TABLE IF NOT EXISTS applications
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  user_id INTEGER NOT NULL,
                  job_description TEXT NOT NULL,
                  resume_text TEXT NOT NULL,
                  status TEXT DEFAULT 'pending',
                  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                  FOREIGN KEY (user_id) REFERENCES users(id))''')
    
    # Create table for storing resume analysis results
    c.execute('''CREATE TABLE IF NOT EXISTS resume_analysis
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  user_id INTEGER DEFAULT 1,
                  job_description TEXT,
                  analysis_date TIMESTAMP,
                  resume_text TEXT)''')
    
    conn.commit()
    conn.close()

def save_application(user_id, job_description, resume_text):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('''INSERT INTO applications 
                 (user_id, job_description, resume_text)
                 VALUES (?, ?, ?)''',
              (user_id, job_description, resume_text))
    conn.commit()
    conn.close()

def get_all_applications():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('''SELECT a.*, u.username 
                 FROM applications a
                 JOIN users u ON a.user_id = u.id
                 ORDER BY a.created_at DESC''')
    applications = c.fetchall()
    conn.close()
    return applications

def update_application_status(application_id, status):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('UPDATE applications SET status = ? WHERE id = ?',
              (status, application_id))
    conn.commit()
    conn.close()

def get_user_profile(user_id):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('SELECT * FROM user_profiles WHERE user_id = ?', (user_id,))
    profile = c.fetchone()
    conn.close()
    return profile

def save_user_profile(user_id, profile_data):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    
    # Check if profile exists
    c.execute('SELECT id FROM user_profiles WHERE user_id = ?', (user_id,))
    profile_exists = c.fetchone()
    
    if profile_exists:
        # Update existing profile
        c.execute('''UPDATE user_profiles SET
                    full_name = ?,
                    email = ?,
                    phone = ?,
                    location = ?,
                    linkedin_url = ?,
                    skills = ?,
                    experience = ?,
                    education = ?,
                    resume_path = ?,
                    updated_at = CURRENT_TIMESTAMP
                    WHERE user_id = ?''',
                 (*profile_data, user_id))
    else:
        # Create new profile
        c.execute('''INSERT INTO user_profiles
                    (user_id, full_name, email, phone, location, linkedin_url,
                     skills, experience, education, resume_path)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                 (user_id, *profile_data))
    
    conn.commit()
    conn.close()

def create_tables():
    """Create necessary tables if they don't exist."""
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    
    # Create table for storing job descriptions
    c.execute('''CREATE TABLE IF NOT EXISTS job_descriptions
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  job_description TEXT,
                  analysis_date TIMESTAMP)''')
    
    conn.commit()
    conn.close()

def save_job_description(job_description):
    """Save job description to database."""
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    
    try:
        # Insert new job description
        c.execute('''INSERT INTO job_descriptions (job_description, analysis_date)
                     VALUES (?, ?)''', (job_description, datetime.now()))
        conn.commit()
        return True
    except Exception as e:
        print(f"Error saving job description: {e}")
        return False
    finally:
        conn.close()

def get_latest_job_description():
    """Retrieve the latest job description."""
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    
    try:
        c.execute('''SELECT job_description FROM job_descriptions 
                     ORDER BY analysis_date DESC 
                     LIMIT 1''')
        result = c.fetchone()
        return result[0] if result else None
    except Exception as e:
        print(f"Error retrieving job description: {e}")
        return None
    finally:
        conn.close()

# Export the functions
__all__ = ['create_tables', 'save_job_description', 'get_latest_job_description'] 