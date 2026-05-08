The following repository contains the source code for the project. It is organized into several directories, each serving a specific purpose:
- `api/`: Contains the code for the API endpoints and related logic.
- `models/`: Contains the data models and database schema definitions.
- `data`: Contains sample data and scripts for populating the database.
- `public/`: Contains static files such as images, stylesheets, and JavaScript files.
- `frontend/`: Contains the code for the frontend application, including components and views.
- `backend/`: Contains the code for the backend application, including server setup and business logic.

Summary of the project:
This project is a web application designed to provide users with a platform to identify and easily find Vex V5 Parts in the Vex Robotics ecosystem. The application allows users to search for specific parts, view detailed information about each part, and access resources for purchasing or learning more about the parts. The goal is to create a user-friendly interface that simplifies the process of finding and understanding Vex V5 Parts for robotics enthusiasts and educators. Each part has a location tag that indicates where it can be found in the physical world, making it easier for users to locate the parts they need. The application also includes features such as user accounts, part reviews, and a community forum for discussing Vex Robotics topics.

## Installation and Setup
To set up the project locally, follow these steps:
1. Clone the repository:
   ```   git clone https://github.com/HPOE-Grimes/HPOEPartFinder.git        
   ``` 
2. Navigate to the project directory:
   ```   cd HPOEPartFinder        
   ```
3. Install the dependencies for both the frontend and backend:
   - For the backend:
     ```   cd backend
     npm install
     ```
   - For the frontend:
     ```   cd ../frontend
     npm install
     ```
4. The database is a ```.csv``` file located in the `data/` directory. You can use this file to populate your database with sample data. You can use a script or a database management tool to import the data from the CSV file into your database.
5. Start the backend server:
   ```   cd ../backend
   python app.py   
   ```
6. Start the frontend development server:
   ```   cd ../frontend
   npm start
   ```
7. Open your web browser and navigate to `http://localhost:5500` to access the application.

## Contributing
Contributions to this project are welcome! If you have any ideas for improvements or new features, please feel free to submit a pull request. Make sure to follow the existing code style and include tests for any new functionality you add. **This only applies to students in the Del Norte High School Honors Principles of Engineering class.**




