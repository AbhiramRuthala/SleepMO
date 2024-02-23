def sleepHTMLGenerator():
    if SleepSchedule == "10 hours":
        myhtml="""
        <h1> 10 hours! That would help out a lot </h1>
        <p> Make sure to use good sleep practices so that you feel better. 10 hours will help you feel replenished. However, there is a good chance that you could wake up earlier.</p>
        """
        report.add_html(title=name + "'s Personalized Sleep Recommendations", html=myhtml)
        report.save("report_add_html.html", overwrite=True)
        if age == "18":
            myhtml = """
                <h1> Great Job </h1>
                <p> This will help with sleep immensely. Stay consistent and continue to endure growth and development. </p>
                """
            report.add_html(html=myhtml, title="data")
            report.save("report_add_html.html", overwrite=True)
        elif age > "18" and age <"27":
            myhtml="""
            <h1> Keep Doing What You're Doing </h1>
            <p> 10 hours is probably a little too much, but it helps with growth. </p>
            <p></p>
            <p> As you'll be wrapping up your growth within the ages of 18-27, this is the time where you shouldn't compromise your growth. Keep doing what you're doing. </p>
            """
            report.add_html(html=myhtml, title="data")
            report.save("report_add_html.html", overwrite=True)
        elif age > "30" and age <"45":
            myhtml="""
            <h1> Wow </h1>
            <p> That's a great amount of sleep. Great job!
            """
            report.add_html(html=myhtml, title="data")
            report.save("report_add_html.html", overwrite=True)
    elif SleepSchedule == "9 hours":
        myhtml="""
        <h1> 9 hours is a good schedule to accomodate to! </h1>
        <p> We will help you every step of the way! </p>
        """
        report.add_html(title=name+"'s Personalized Sleep Recommendations", html=myhtml)
        report.save("report_add_html.html", overwrite=True)
        if age == "18":
            myhtml = """
                <h1> Good Job </h1>
                <p> This will help with sleep immensely. Stay consistent and continue to endure growth and development. </p>
                """
            report.add_html(html=myhtml, title="data")
            report.save("report_add_html.html", overwrite=True)
        elif age > "18" and age <"27":
            myhtml="""
            <h1> Keep Doing What You're Doing </h1>
            <p> 9 hours is a great amount of sleep </p>
            <p></p>
            <p> As you'll be wrapping up your growth within the ages of 18-27, this is the time where you shouldn't compromise your growth. Keep doing what you're doing. </p>
            """
            report.add_html(html=myhtml, title="data")
            report.save("report_add_html.html", overwrite=True)
        elif age > "30" and age <"45":
            myhtml="""
            <h1> Wow </h1>
            <p> That's a good amount of sleep. Great job!
            """
            report.add_html(html=myhtml, title="data")
            report.save("report_add_html.html", overwrite=True)
    elif SleepSchedule == "8 hours":
        myhtml="""
        <h1> 8 Hours! Great Job </h1>
        <p> A lot of people lack consistency, so the best thing that I can recommend for you is to stay consistent with your sleep schedule. </p>
        """
        report.add_html(title=name + "'s Personalized Sleep Recommendations", html=myhtml)
        report.save("report_add_html.html", overwrite=True)
        if age == "18":
            myhtml="""
            <h1> Sleep Recommendations </h1>
            <p> You're still growing, so make sure to stay consistent and endure the growth.</p>
            """
            report.add_html(html=myhtml, title="Data")
            report.save("report_add_html.html", overwrite=True)
        elif age > 18 and age < 27:
            myhtml="""
            <h1> Sleep Recommendations </h1>
            <p> Main thing is to stay consistent. Also, make sure to practice good routine practices when you go to sleep. </p>
            """
            report.add_html(html=myhtml, title="Data")
            report.save("report_add_html.html", overwrite=True)
        elif age > 30 and age < 45:
            myhtml = """
            <h1> Sleep Recommendations </h1>
            <p> Main thing is to stay consistent. Also, make sure to practice good routine practices when you go to sleep. </p>
            """
            report.add_html(html=myhtml, title="Data")
            report.save("report_add_html.html", overwrite=True)
    elif SleepSchedule == "7 hours":
        if age == "18":
            myhtml="""
            <p> Keep yourself safe. </p>
            """
            report.add_html(html=myhtml, title="Data")
            report.save("report_add_html.html", overwrite=True)
        elif age > "18" and age < "27":
            myhtml="""
            <p> Make sure to sleep at consistent times so that your growth is correct. </p>
            """
            report.add_html(html=myhtml, title=name+"'s Personalized Sleep Recommendations")
            report.save("report_add_html.html", overwrite=True)
    elif SleepSchedule == "7 hours":
        myhtml="""
        <h1> Aim for more sleep</h1>
        """
        report.add_html(html=myhtml, title="data")
        report.save("report_add_html.html", overwrite=True)
        if age == "18":
            myhtml="""
            <h1> 7 hours? Let's see if you can get more </h1>
            <p> If you're able to carve out some time, try to increase your sleep by just a little bit. Make sure that you stay consistent as well.</p>"""
            report.add_html(html=myhtml, title="data")
            report.save("report_add_html.html", overwrite=True)
        elif age > "18" and age < "27":
            myhtml = """
            <h1> 7 hours? Let's see if you can get more </h1>
            <p> If you're able to carve out some time, try to increase your sleep by just a little bit. Make sure that you stay consistent as well.</p>"""
            report.add_html(html=myhtml, title="data")
            report.save("report_add_html.html", overwrite=True)
        elif age > "30" and age < "45":
            myhtml = """
            <h1> 7 hours? Let's see if you can get more </h1>
            <p> If you're able to carve out some time, try to increase your sleep by just a little bit. Make sure that you stay consistent as well.</p>"""
            report.add_html(html=myhtml, title="data")
            report.save("report_add_html.html", overwrite=True)
    elif SleepSchedule == "6 hours":
        myhtml="""
        <h1> You should aim for more sleep </h1>
        <p> 6 hours isn't the most sustainable amount of sleep </p>"""
        report.add_html(html=myhtml, title=name+"'s Personalized Sleep Recommendations")
        report.save("report_add_html.html", overwrite=True)
        if age == "18":
            myhtml = """
                <h1> Great Job </h1>
                <p> This will help with sleep immensely. Stay consistent and continue to endure growth and development. </p>
                """
            report.add_html(html=myhtml, title="data")
            report.save("report_add_html.html", overwrite=True)
        elif age > "18" and age <"27":
            myhtml="""
            <h1> Keep Doing What You're Doing </h1>
            <p> 10 hours is probably a little too much, but it helps with growth. </p>
            <p></p>
            <p> As you'll be wrapping up your growth within the ages of 18-27, this is the time where you shouldn't compromise your growth. Keep doing what you're doing. </p>
            """
            report.add_html(html=myhtml, title="data")
            report.save("report_add_html.html", overwrite=True)
        elif age > "30" and age <"45":
            myhtml="""
            <h1> Wow </h1>
            <p> That's a great amount of sleep. Great job!
            """
            report.add_html(html=myhtml, title="data")
            report.save("report_add_html.html", overwrite=True)
    elif SleepSchedule == "5 hours":
        myhtml="""
        <h1> Might need to select a higher schedule </h1>
        """
        report.add_html(html=myhtml, title=name+"'s Personalized Sleep Recommendations")
        report.save("report_add_html.html", overwrite=True)
        if age == "18":
            myhtml="""
            <h1> Try to aim for more sleep </h1>
            """
            report.add_html(html=myhtml, title="data")
            report.save("report_add_html.html", overwrite=True)
        elif age > "18" and age <"27":
            myhtml="""
            <h1> At your age, the main goal would be to aim for more sleep </h1>
            """
            report.add_html(html=myhtml, title="data")
            report.save("report_add_html.html", overwrite=True)
        elif age > "30" and age < "45":
            myhtml="""
            <h1> At your age, the main goal would be to aim for more sleep </h1>
            """
            report.add_html(html=myhtml, title="data")
            report.save("report_add_html.html", overwrite=True)
    else:
        pass
