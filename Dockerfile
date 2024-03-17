FROM python:3.11

# Add a non-root user and switch to it
RUN useradd -m -u 1000 user
USER user

# Set environment variables
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Set the working directory
WORKDIR $HOME/app

# Clone the Git repository
RUN git clone https://github.com/brlrb/chatpdf.git $HOME/app

# Assuming you want to install requirements from the cloned repo
COPY --chown=user: ./requirements.txt $HOME/app/requirements.txt
RUN pip install --user -r requirements.txt

# Copy the rest of your application (if needed)
COPY --chown=user: . .

# Command to run your application
CMD ["chainlit", "run", "app.py", "--port", "7860"]