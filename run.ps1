Remove-Item -Recurse -Force .git

git init
git branch -M main

@"
dataset/
*.csv
"@ | Set-Content .gitignore

git add .
git commit -m "Initial commit"

git remote add origin https://github.com/Amancs1541/xyxz.git
git push -u origin main