function getCookie(name) {
    const cookieValue = document.cookie
        .split(";")
        .map((cookie) => cookie.trim())
        .find((cookie) => cookie.startsWith(name + "="));

    if (cookieValue) {
        return decodeURIComponent(cookieValue.split("=")[1]);
    } else {
        return null;
    }
}

function openModal(name) {
    var id_name = "#" + name
    $(id_name).modal("toggle");
}

function closeModal(name) {
    var id_name = "#" + name
    $(id_name).modal("hide");
}

function buildUrl(baseUrl, params) {
    const queryString = new URLSearchParams(params).toString();
    return queryString ? `${baseUrl}?${queryString}` : baseUrl;
}

function ensurePageParam(url, params = {}) {
    const urlObj = new URL(url);

    // Append existing params to the URL
    Object.entries(params).forEach(([key, value]) => {
        urlObj.searchParams.set(key, value);
    });

    // Ensure 'page' param exists, default to 1 if missing
    if (!urlObj.searchParams.has('page')) {
        urlObj.searchParams.set('page', '1');
    }

    return urlObj.toString();
}

function initializeModels(instance, objects = []) {
    // Default Vue

    instance.lists = {}
    instance.url_models = {}
    instance.loading_state = {}

    // Ensure lists, url_models, and loading_state are reactive
    objects.forEach(element => {
        Vue.set(instance.lists, element.toUpperCase(), []);
        Vue.set(instance.url_models, element.toUpperCase(), `/${element.toLowerCase()}/`);
        Vue.set(instance.loading_state, element.toUpperCase(), true);
    });

    instance.endpoints = {
        'LIST_GET': '/api/list-get',
        'LIST_CREATE': '/api/list-create',
        'GET_UPDATE_DESTROY': '/api/get-update-destroy'
    }

    instance.info = { 'MESSAGE': '', 'TITLE': '' }

    return instance

}

async function universalGet(object,
    endpoint_name,
    model_name,
    list_item,
    loading_state = null,
    filters = null, custom_url = ''
) {

    var url = custom_url
        ? custom_url
        : object.endpoints[endpoint_name] + object.url_models[model_name];

    const authToken = localStorage.getItem('token')
    let result = null

    try {

        if (loading_state !== null) {
            object.loading_state[loading_state] = true
        }

        if (filters !== null) {
            url = buildUrl(url, filters)
        }

        result = await axios.get(url, {
            headers: {
                'Authorization': `Bearer ${authToken}`
            }
        });

        if (result) {
            object.lists[list_item] = result.data
        }

    } catch (error) {
        console.log(error)
    }

    object.loading_state[loading_state] = false
    return result

}

async function universalPatch(object,
    endpoint_name,
    model_name,
    list_item,
    loading_state = false, identifier = null, props = null
) {
    console.log(props)
    const url = object.endpoints[endpoint_name]
        + object.url_models[model_name]
        + object.state[identifier] + "/"

    try {
        axios.defaults.headers.common['X-CSRFToken'] = getCookie('csrftoken')
        const result = await axios.patch(url, object.props[props])
        if (result) {
            const get_result = await universalGet(object, 'LIST_GET', model_name, list_item, loading_state, { page: 1 })
        }
        return result
    } catch (error) { console.log(error) }

}

async function universalPost(object,
    endpoint_name,
    model_name,
    list_item,
    loading_state = null, prop
) {

    try {
        const url = object.endpoints[endpoint_name] + object.url_models[model_name]
        axios.defaults.headers.common['X-CSRFToken'] = getCookie('csrftoken')
        const result = await axios.post(url, prop)
        return result

    } catch (error) { }

}

async function universalDelete(object,
    endpoint_name,
    model_name,
    list_item,
    loading_state = false, identifier = null
) {

    const url = object.endpoints[endpoint_name]
        + object.url_models[model_name]
        + object.state[identifier] + "/"

    try {

        axios.defaults.headers.common['X-CSRFToken'] = getCookie('csrftoken')
        const result = await axios.delete(url)

        if (result) {
            const get_result = await universalGet(object, 'LIST_GET', model_name, list_item, loading_state, { page: 1 })
        }
        return result

    } catch (error) { console.log(error) }

}