import asyncio
import edge_tts

async def generate():
    text = "वक्रतुण्ड महाकाय सूर्यकोटि समप्रभ निर्विघ्नं कुरु मे देव सर्वकार्येषु सर्वदा"
    
    communicate = edge_tts.Communicate(
        text,
        voice="hi-IN-SwaraNeural",
        rate="-15%"  
    )

    await communicate.save("ganesha_shloka.mp3")

asyncio.run(generate())