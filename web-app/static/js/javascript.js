$(document).ready(function()
{
    // $('#predict').prop('disabled',false);
    $('#predict').click(function()
    {
        console.log("Here")

        event.preventDefault();
        $('#result p').text('Please wait...')
        var form_data = new FormData($('#formId')[0]);
        $.ajax({
            type: 'POST',
            url: '/predict',
            data: form_data,
            contentType: false,
            cache: false,
            processData: false,
            success: function(data)
            {
                console.log(data);
                $('#result p').text(data);
                // $('#predict').prop('disabled',true);
            }
        })
    });
    
});