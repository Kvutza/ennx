if [ -n "${PIXI_PROJECT_ROOT:-}" ]; then
    case ":${PATH}:" in
        *":$PIXI_PROJECT_ROOT/tools:"*) ;;
        *) export PATH="$PIXI_PROJECT_ROOT/tools:$PATH" ;;
    esac
fi
