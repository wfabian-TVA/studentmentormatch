import streamlit as st
import pandas as pd
import numpy as np
import math
import base64
from io import BytesIO

st.set_page_config(page_title="Speed Networking Matcher", layout="wide")

st.title("Speed Networking Matcher")
st.write("""
This app helps you match students with mentors for speed networking sessions.
Each student will have 3 conversations with different mentors, and each mentor
will have no more than 5 conversations per day.
""")

# Session state initialization
if 'schedule_generated' not in st.session_state:
    st.session_state.schedule_generated = False
if 'all_students_matched' not in st.session_state:
    st.session_state.all_students_matched = False

# Helper functions
def calculate_match_score(student_interests, mentor_interests):
    """Calculate matching score between a student and mentor based on interests"""
    # Convert interests to sets and find intersection
    student_set = set(interest.strip().lower() for interest in student_interests if interest)
    mentor_set = set(interest.strip().lower() for interest in mentor_interests if interest)
    
    # Count matching interests
    matching = student_set.intersection(mentor_set)
    
    if not student_set or not mentor_set:
        return 0
    
    # Jaccard similarity: intersection / union
    score = len(matching) / math.sqrt(len(student_set) * len(mentor_set))
    return score

def generate_matches(students_df, mentors_df, time_slots):
    """Generate optimal matches between students and mentors"""
    
    # Create lists to store all potential meetings
    potential_meetings = []
    
    # Create compatibility matrix
    for s_idx, student in students_df.iterrows():
        student_interests = [student[col] for col in students_df.columns if 'interest' in col.lower()]
        
        for m_idx, mentor in mentors_df.iterrows():
            mentor_interests = [mentor[col] for col in mentors_df.columns if 'interest' in col.lower()]
            
            # Calculate match score
            score = calculate_match_score(student_interests, mentor_interests)
            
            # Add all possible time slot combinations
            for time_slot in time_slots:
                potential_meetings.append({
                    'student_id': student['StudentID'],
                    'student_name': student['Name'],
                    'mentor_id': mentor['MentorID'],
                    'mentor_name': mentor['Name'],
                    'time_slot': time_slot,
                    'score': score
                })
    
    # Convert to DataFrame and sort by score
    meetings_df = pd.DataFrame(potential_meetings)
    meetings_df = meetings_df.sort_values('score', ascending=False)
    
    # Initialize tracking variables
    assignments = []
    student_meeting_counts = {sid: 0 for sid in students_df['StudentID']}
    mentor_meeting_counts = {mid: 0 for mid in mentors_df['MentorID']}
    student_schedule = {sid: [] for sid in students_df['StudentID']}
    mentor_schedule = {mid: [] for mid in mentors_df['MentorID']}
    student_met_mentors = {sid: [] for sid in students_df['StudentID']}
    
    # Make assignments
    for _, meeting in meetings_df.iterrows():
        s_id = meeting['student_id']
        m_id = meeting['mentor_id']
        t_slot = meeting['time_slot']
        
        # Check all constraints
        if (student_meeting_counts[s_id] < 3 and  # Student needs more meetings
            mentor_meeting_counts[m_id] < 5 and   # Mentor hasn't exceeded max
            t_slot not in student_schedule[s_id] and  # Student is available
            t_slot not in mentor_schedule[m_id] and   # Mentor is available
            m_id not in student_met_mentors[s_id]):   # Student hasn't met this mentor
            
            # Add the assignment
            assignments.append(meeting)
            
            # Update trackers
            student_meeting_counts[s_id] += 1
            mentor_meeting_counts[m_id] += 1
            student_schedule[s_id].append(t_slot)
            mentor_schedule[m_id].append(t_slot)
            student_met_mentors[s_id].append(m_id)
    
    # Convert to DataFrame
    schedule_df = pd.DataFrame(assignments)
    
    # Check if all students got 3 meetings
    all_matched = all(count == 3 for count in student_meeting_counts.values())
    
    return schedule_df, all_matched, student_meeting_counts

def generate_sample_data():
    """Generate sample data for testing"""
    # Generate 10 students with interests
    students = []
    interests = ["Data Science", "Machine Learning", "Web Development", 
                "Mobile Apps", "Cloud Computing", "Cybersecurity", 
                "UI/UX Design", "Product Management", "Entrepreneurship", 
                "Artificial Intelligence", "Blockchain", "IoT"]
    
    for i in range(1, 11):
        # Randomly select 3-5 interests
        num_interests = np.random.randint(2, 6)
        student_interests = np.random.choice(interests, num_interests, replace=False)
        
        student = {
            'StudentID': f'S{i:03}',
            'Name': f'Student {i}',
        }
        
        # Add interests columns
        for j in range(5):  # Max 5 interest columns
            if j < len(student_interests):
                student[f'Interest{j+1}'] = student_interests[j]
            else:
                student[f'Interest{j+1}'] = ''
        
        students.append(student)
    
    # Generate 5 mentors with interests
    mentors = []
    for i in range(1, 6):
        # Randomly select 3-5 interests
        num_interests = np.random.randint(2, 6)
        mentor_interests = np.random.choice(interests, num_interests, replace=False)
        
        mentor = {
            'MentorID': f'M{i:03}',
            'Name': f'Mentor {i}',
        }
        
        # Add interests columns
        for j in range(5):  # Max 5 interest columns
            if j < len(mentor_interests):
                mentor[f'Interest{j+1}'] = mentor_interests[j]
            else:
                mentor[f'Interest{j+1}'] = ''
        
        mentors.append(mentor)
    
    return pd.DataFrame(students), pd.DataFrame(mentors)

def get_excel_download_link(df, filename, sheet_name):
    """Generate a download link for dataframe as Excel file"""
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    df.to_excel(writer, sheet_name=sheet_name, index=False)
    writer.close()
    excel_data = output.getvalue()
    b64 = base64.b64encode(excel_data).decode()
    href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="{filename}.xlsx">Download {filename}</a>'
    return href

# Sidebar for data input
st.sidebar.header("Input Data")

data_source = st.sidebar.radio(
    "Select data source",
    ["Upload Files", "Use Sample Data"]
)

if data_source == "Upload Files":
    students_file = st.sidebar.file_uploader("Upload Students CSV", type=["csv"])
    mentors_file = st.sidebar.file_uploader("Upload Mentors CSV", type=["csv"])
    
    if students_file is not None and mentors_file is not None:
        students_df = pd.read_csv(students_file)
        mentors_df = pd.read_csv(mentors_file)
        st.sidebar.success("Files uploaded successfully!")
    else:
        st.sidebar.info("Please upload both files to continue.")
        students_df = None
        mentors_df = None
else:
    # Generate sample data
    students_df, mentors_df = generate_sample_data()
    st.sidebar.success("Sample data generated!")
    
    # Show download links for the templates
    st.sidebar.markdown("### Download Sample Data")
    st.sidebar.markdown(get_excel_download_link(students_df, "students_template", "Students"), unsafe_allow_html=True)
    st.sidebar.markdown(get_excel_download_link(mentors_df, "mentors_template", "Mentors"), unsafe_allow_html=True)

# Time slots input
st.sidebar.header("Time Slots")
start_time = st.sidebar.time_input("Start time", value=pd.to_datetime('16:30').time())
end_time = st.sidebar.time_input("End time", value=pd.to_datetime('18:30').time())
slot_duration = st.sidebar.number_input("Session duration (minutes)", min_value=5, max_value=30, value=10)
buffer_time = st.sidebar.number_input("Buffer time between sessions (minutes)", min_value=0, max_value=15, value=5)

# Input validation
if start_time >= end_time:
    st.sidebar.error("End time must be after start time.")
else:
    # Generate time slots (with buffer time)
    time_slots = []
    formatted_time_slots = []
    current_time = pd.to_datetime(start_time.strftime('%H:%M'))
    end_datetime = pd.to_datetime(end_time.strftime('%H:%M'))
    
    while current_time < end_datetime:
        # Add the slot start time
        slot_start = current_time.strftime('%H:%M')
        
        # Calculate slot end time (for display purposes)
        slot_end = (current_time + pd.Timedelta(minutes=slot_duration)).strftime('%H:%M')
        
        # Create a formatted time slot string
        formatted_slot = f"{slot_start} - {slot_end}"
        
        time_slots.append(slot_start)
        formatted_time_slots.append(formatted_slot)
        
        # Move to next slot (including buffer time)
        current_time += pd.Timedelta(minutes=slot_duration + buffer_time)

# Generate matches button
if students_df is not None and mentors_df is not None:
    if st.sidebar.button("Generate Matches"):
        # Display the input data
        with st.expander("Students Data"):
            st.dataframe(students_df)
        
        with st.expander("Mentors Data"):
            st.dataframe(mentors_df)
        
        with st.expander("Time Slots"):
            # Display time slots with duration information
            st.write(f"Session duration: {slot_duration} minutes with {buffer_time} minute buffer")
            st.write(formatted_time_slots)
        
        # Run the matching algorithm
        with st.spinner("Generating optimal matches..."):
            # Use the time slots for the algorithm
            schedule_df, all_matched, student_meeting_counts = generate_matches(students_df, mentors_df, time_slots)
            
            # Set session state
            st.session_state.schedule_generated = True
            st.session_state.all_students_matched = all_matched
            
            # Save results to session state
            st.session_state.schedule_df = schedule_df
            st.session_state.student_meeting_counts = student_meeting_counts

# Display results if available
if st.session_state.schedule_generated:
    st.header("Matching Results")
    
    if not st.session_state.all_students_matched:
        st.warning("⚠️ Not all students received 3 meetings. You may need more mentors or time slots.")
        
        # Show student meeting counts
        meeting_counts_df = pd.DataFrame({
            'StudentID': list(st.session_state.student_meeting_counts.keys()),
            'Meetings Assigned': list(st.session_state.student_meeting_counts.values())
        })
        meeting_counts_df = meeting_counts_df.sort_values('Meetings Assigned')
        
        st.subheader("Students with Incomplete Schedules")
        incomplete_students = meeting_counts_df[meeting_counts_df['Meetings Assigned'] < 3]
        st.dataframe(incomplete_students)
    
    # Display the schedule
    st.subheader("Complete Schedule")
    
    # Sort by time slot, then student
    sorted_schedule = st.session_state.schedule_df.sort_values(['time_slot', 'student_name'])
    
    # Format for display
    display_schedule = sorted_schedule[['time_slot', 'student_name', 'mentor_name', 'score']]
    display_schedule = display_schedule.rename(columns={
        'time_slot': 'Time',
        'student_name': 'Student',
        'mentor_name': 'Mentor',
        'score': 'Match Score'
    })
    
    # Format score as percentage
    display_schedule['Match Score'] = display_schedule['Match Score'].apply(lambda x: f"{x:.1%}")
    
    st.dataframe(display_schedule)
    
    # Student view
    st.subheader("Student Schedules")
    student_schedule = sorted_schedule.pivot_table(
        index='student_name',
        columns='time_slot',
        values='mentor_name',
        aggfunc='first'
    ).reset_index()
    
    st.dataframe(student_schedule)
    
    # Mentor view
    st.subheader("Mentor Schedules")
    mentor_schedule = sorted_schedule.pivot_table(
        index='mentor_name',
        columns='time_slot',
        values='student_name',
        aggfunc='first'
    ).reset_index()
    
    st.dataframe(mentor_schedule)
    
    # Download links
    st.subheader("Download Results")
    st.markdown(get_excel_download_link(display_schedule, "complete_schedule", "Schedule"), unsafe_allow_html=True)
    st.markdown(get_excel_download_link(student_schedule, "student_schedules", "Student Schedules"), unsafe_allow_html=True)
    st.markdown(get_excel_download_link(mentor_schedule, "mentor_schedules", "Mentor Schedules"), unsafe_allow_html=True)

# Instructions at the bottom
with st.expander("Instructions"):
            st.write("""
    ### How to use this app:
    
    1. **Input data**: Upload CSV files with student and mentor information, or use the sample data.
    2. **Set time slots**: The app is configured for sessions from 4:30 PM to 6:30 PM with 10-minute conversations and 5-minute buffer times between sessions.
    3. **Generate matches**: Click the button to create optimal matches based on shared interests.
    4. **Review results**: Check the generated schedules and download as Excel files.
    
    ### Required CSV format:
    
    **Students CSV**:
    - `StudentID`: Unique identifier for each student
    - `Name`: Student's name
    - `Interest1`, `Interest2`, etc.: Columns containing student interests
    
    **Mentors CSV**:
    - `MentorID`: Unique identifier for each mentor
    - `Name`: Mentor's name
    - `Interest1`, `Interest2`, etc.: Columns containing mentor interests
    
    You can download the sample data to see the expected format.
    """)
