import asyncio
import hud

async def test():
    from env import init, env
    await init()

    task = env('tau2',
        domain='telecom',
        task_id='[mobile_data_issue]airplane_mode_on|user_abroad_roaming_enabled_off[PERSONA:None]',
        task_split='base'
    )

    async with hud.eval(task) as ctx:
        print('✓ Telecom scenario initialized successfully')
        await ctx.submit('')

asyncio.run(test())
