import gymnasium

gymnasium.register(
    id="Go2Walk-v0",
    entry_point=f"{__name__}.go2_walk_env:Go2Env"
)