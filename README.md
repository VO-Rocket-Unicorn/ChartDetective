<h1> Chart Simulator  </h1>





**1.** Clone the forked repository.

```terminal
git clone https://github.com/VO-Rocket-Unicorn/ChartDetective
```

**2.** Navigate to the project directory.

```terminal
cd ChartDetective
```

**3.** Add a reference(remote) to the original repository.

```
git remote add upstream https://github.com/VO-Rocket-Unicorn/ChartDetective
```

**4.** Check the remotes for this repository.
```
git remote -v
```

**5.** Always take a pull from the upstream repository to your master branch to keep it at par with the main project(updated repository).

```
git pull upstream main
```
**6.** Install Dependencies

Frontend
```
cd frontend
npm Install
```
Backend
```
cd backend

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
pip install -r requirements.txt

```
**7.** Run the project

Frontend
```
cd frontend
npm start
```
Backend 
```
uvicorn curve_api:app --reload
```

**8.**  MAKE NECESSARY CHANGES IN THE PROJECT TO EDIT IT
<br>


**9.** Create a new branch.

```terminal
git checkout -b <your_branch_name>
```

**10.** Add & Commit your changes.

```terminal
  git add .
  git commit -m "<your_commit_message>"
```

**11.** Push your local branch to the remote repository.

```terminal
git push -u origin <your_branch_name>
```

**12.** Create a Pull Request!
